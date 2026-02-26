from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

import json
import math
import random
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import make_expanding_time_folds, parse_and_sort_train, parse_predictions
from src.ratings import apply_massey_features, elo_expected_score, elo_mov_multiplier, fit_massey_ridge

SEED = 23
ROOT = Path(__file__).resolve().parent
INITIAL_ELO = 1500.0
K_BASE = 24.0
HOME_ADV = 50.0
ALPHA_GRID = [0.1, 1.0, 10.0, 50.0, 100.0]
A_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
G_GRID = [50, 100, 150]
TAU_GRID = [25, 50, 75]


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


def rmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.sqrt(np.mean((y-p)**2)))


def mae(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean(np.abs(y-p)))


def ridge_pipe(alpha):
    return Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=float(alpha), random_state=SEED))])


def ensure_root_inputs(root: Path):
    train = root / 'Train.csv'
    if not train.exists():
        raise FileNotFoundError(train)
    def ensure(name):
        p = root / name
        if p.exists():
            return p
        for c in [root/'Submission.zip'/name, root/'Submission.zip1'/name, root/'Submission.zip2'/name]:
            if c.exists():
                shutil.copy2(c, p)
                print(f'Copied missing root input {name} from {c}')
                return p
        raise FileNotFoundError(name)
    return train, ensure('Predictions.csv'), ensure('Rankings.xlsx')


def detect_seasons(date_series, gap_days=60):
    d = pd.to_datetime(date_series).reset_index(drop=True)
    n = len(d)
    sid = np.zeros(n, dtype=int)
    cur = 0
    for i in range(1, n):
        if (pd.Timestamp(d.iloc[i]) - pd.Timestamp(d.iloc[i-1])).days > gap_days:
            cur += 1
        sid[i] = cur
    g = np.zeros(n, dtype=int)
    slen = {}
    for s in np.unique(sid):
        idx = np.flatnonzero(sid == s)
        g[idx] = np.arange(len(idx), dtype=int)
        slen[int(s)] = int(len(idx))
    return sid, g, slen


def k_mult(cfg, g, season_len):
    A = max(float(cfg['A']), 0.0)
    if A <= 0:
        return 1.0
    if cfg['decay_type'] == 'linear':
        Ge = max(1, min(int(cfg['G']), int(season_len)))
        return 1.0 + A * max(0.0, 1.0 - float(g)/float(Ge))
    tau = max(1.0, float(cfg['tau']))
    return 1.0 + A * math.exp(-float(g)/tau)


def run_elo_train(train_df, season_ctx, home_adv, cfg):
    sid, gidx, slen = season_ctx
    ratings = {}
    rows = []
    for i, row in enumerate(train_df.itertuples(index=False)):
        hid, aid = int(row.HomeID), int(row.AwayID)
        h = float(ratings.get(hid, INITIAL_ELO)); a = float(ratings.get(aid, INITIAL_ELO))
        diff_ha = h - a + float(home_adv)
        p = float(elo_expected_score(diff_ha))
        s = int(sid[i]) if len(sid) else 0
        g = int(gidx[i]) if len(gidx) else i
        km = k_mult(cfg, g, slen.get(s, len(train_df)))
        rows.append({'elo_home_pre':h,'elo_away_pre':a,'elo_diff_pre':diff_ha,'elo_prob_home_pre':p,'elo_k_mult':km,'elo_season_id':s,'elo_games_into_season':g})
        m = float(row.HomeWinMargin)
        score = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        dlt = K_BASE * km * float(elo_mov_multiplier(m, diff_ha)) * (score - p)
        ratings[hid] = h + dlt; ratings[aid] = a - dlt
    return pd.DataFrame(rows, index=train_df.index), ratings


def run_elo_derby(pred_df, final_ratings):
    e1 = pred_df['Team1_ID'].astype(int).map(final_ratings).fillna(INITIAL_ELO).astype(float)
    e2 = pred_df['Team2_ID'].astype(int).map(final_ratings).fillna(INITIAL_ELO).astype(float)
    diff = e1 - e2
    return pd.DataFrame({'elo_home_pre':e1,'elo_away_pre':e2,'elo_diff_pre':diff,'elo_prob_home_pre':np.asarray(elo_expected_score(diff.to_numpy()), float)}, index=pred_df.index)

def conf_dummies(df, c1, c2):
    h = pd.get_dummies(df[c1].astype(str), prefix='conf_home')
    a = pd.get_dummies(df[c2].astype(str), prefix='conf_away')
    out = pd.concat([h.reset_index(drop=True), a.reset_index(drop=True)], axis=1)
    out.index = df.index
    return out


def build_fold_caches(train_df, folds):
    team_ids = sorted(set(train_df['HomeID'].astype(int)) | set(train_df['AwayID'].astype(int)))
    caches = []
    for f in folds:
        tr_idx = np.asarray(f['train_idx'], int); va_idx = np.asarray(f['val_idx'], int)
        fit_games = train_df.iloc[tr_idx].copy(); tr = train_df.iloc[tr_idx].copy(); va = train_df.iloc[va_idx].copy()
        massey = fit_massey_ridge(fit_games, team_ids=team_ids, alpha=30.0)
        tr_m = apply_massey_features(tr, massey, 'HomeID', 'AwayID', neutral_site=False)[['massey_diff']]
        va_m = apply_massey_features(va, massey, 'HomeID', 'AwayID', neutral_site=False)[['massey_diff']]
        Xtr = pd.concat([tr_m.reset_index(drop=True), conf_dummies(tr,'HomeConf','AwayConf').reset_index(drop=True)], axis=1)
        Xva = pd.concat([va_m.reset_index(drop=True), conf_dummies(va,'HomeConf','AwayConf').reset_index(drop=True)], axis=1)
        Xtr.index = tr.index; Xva.index = va.index
        caches.append({'fold':int(f['fold']),'tr_idx':tr_idx,'va_idx':va_idx,'ytr':tr['HomeWinMargin'].to_numpy(float),'yva':va['HomeWinMargin'].to_numpy(float),'Xtr_static':Xtr,'Xva_static':Xva})
    return caches


def build_cfg_grid():
    cfgs = [{'decay_type':'linear','A':0.0,'G':100,'tau':50}]
    for A in A_GRID:
        if A <= 0: continue
        for G in G_GRID:
            cfgs.append({'decay_type':'linear','A':float(A),'G':int(G),'tau':50})
    for A in A_GRID:
        if A <= 0: continue
        for tau in TAU_GRID:
            cfgs.append({'decay_type':'exponential','A':float(A),'G':100,'tau':int(tau)})
    return cfgs


def cfg_key(cfg):
    if cfg['decay_type'] == 'linear':
        return f"linear_A{cfg['A']:.2f}_G{int(cfg['G'])}"
    return f"exponential_A{cfg['A']:.2f}_tau{int(cfg['tau'])}"


def cfg_label(cfg):
    if float(cfg['A']) <= 0: return 'A=0 (baseline)'
    if cfg['decay_type'] == 'linear': return f"linear A={cfg['A']:.2f}, G={cfg['G']}"
    return f"exp A={cfg['A']:.2f}, tau={cfg['tau']}"


def budget_exhausted(b):
    if b.get('stop_reason'): return True
    if time.perf_counter()-b['start'] >= b['max_total']:
        b['stop_reason'] = 'MAX_TOTAL_SECONDS'; return True
    if time.perf_counter()-b['tune_start'] >= b['max_tune']:
        b['stop_reason'] = 'MAX_TUNING_SECONDS'; return True
    if b['fit_count'] >= b['max_fits']:
        b['stop_reason'] = 'MAX_MODEL_FITS'; return True
    return False


def can_fit(b, n=1):
    if budget_exhausted(b): return False
    if b['fit_count'] + n > b['max_fits']:
        b['stop_reason'] = 'MAX_MODEL_FITS'; return False
    return True


def eval_combo(train_df, fold_caches, elo_feat, cfg, alpha, b, stage):
    oof = np.full(len(train_df), np.nan)
    oof_fold = np.full(len(train_df), -1, int)
    for fc in fold_caches:
        if not can_fit(b, 1):
            return None
        Xtr = pd.concat([elo_feat.loc[fc['tr_idx'], ['elo_diff_pre']].reset_index(drop=True), fc['Xtr_static'].reset_index(drop=True)], axis=1)
        Xva = pd.concat([elo_feat.loc[fc['va_idx'], ['elo_diff_pre']].reset_index(drop=True), fc['Xva_static'].reset_index(drop=True)], axis=1)
        Xtr.columns = [str(c) for c in Xtr.columns]; Xva.columns = [str(c) for c in Xva.columns]
        Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
        mdl = ridge_pipe(alpha)
        mdl.fit(Xtr.astype(float), fc['ytr'])
        b['fit_count'] += 1
        yh = np.asarray(mdl.predict(Xva.astype(float)), float)
        oof[fc['va_idx']] = yh
        oof_fold[fc['va_idx']] = int(fc['fold'])
    vi = np.concatenate([fc['va_idx'] for fc in fold_caches])
    if np.isnan(oof[vi]).any():
        return None
    y = train_df.iloc[vi]['HomeWinMargin'].to_numpy(float); p = oof[vi]
    oof_df = pd.DataFrame({'row_index':vi,'fold':oof_fold[vi],'y_true':y,'y_pred':p}).sort_values(['fold','row_index'], kind='mergesort').reset_index(drop=True)
    return {'cfg':cfg.copy(),'alpha':float(alpha),'rmse':rmse(y,p),'mae':mae(y,p),'pred_std':float(np.std(p)),'actual_std':float(np.std(y)),'oof':oof_df,'stage':stage,'n_folds':len(fold_caches)}


def tune_budgeted(train_df, fold_caches, season_ctx, home_adv, b):
    b['tune_start'] = time.perf_counter()
    cfgs = build_cfg_grid(); elo_cache = {}; results = []; seen = set()
    def get_elo(cfg):
        k = cfg_key(cfg)
        if k not in elo_cache:
            elo_cache[k] = run_elo_train(train_df, season_ctx, home_adv, cfg)
        return elo_cache[k]

    stage1_alpha = 10.0
    print(f'Stage 1 tuning: {len(cfgs)} Elo configs at alpha={stage1_alpha}')
    for cfg in cfgs:
        if budget_exhausted(b): break
        elo_feat, _ = get_elo(cfg)
        res = eval_combo(train_df, fold_caches, elo_feat, cfg, stage1_alpha, b, 'stage1')
        if res is None: break
        results.append(res); seen.add((cfg_key(cfg), float(stage1_alpha)))
    if not results:
        raise RuntimeError('No tuning evaluations completed.')

    stage1_k = sorted([r for r in results if r['cfg']['A'] > 0], key=lambda r:(r['rmse'], r['mae']))[:2]
    stage1_b = [r for r in results if r['cfg']['A'] <= 0]
    if not stage1_b:
        raise RuntimeError('Baseline A=0 was not evaluated before tuning budget stop.')
    refine_cfgs = [stage1_b[0]['cfg']] + [r['cfg'] for r in stage1_k if cfg_key(r['cfg']) != cfg_key(stage1_b[0]['cfg'])]
    stage2_alphas = [a for a in ALPHA_GRID if float(a) != stage1_alpha]
    print('Stage 2 tuning refine:', [cfg_label(c) for c in refine_cfgs], 'alphas=', stage2_alphas)
    for cfg in refine_cfgs:
        elo_feat, _ = get_elo(cfg)
        for alpha in stage2_alphas:
            if budget_exhausted(b): break
            k = (cfg_key(cfg), float(alpha))
            if k in seen: continue
            res = eval_combo(train_df, fold_caches, elo_feat, cfg, alpha, b, 'stage2')
            if res is None: break
            results.append(res); seen.add(k)
        if budget_exhausted(b): break

    summary = pd.DataFrame([{ 'config_key':cfg_key(r['cfg']), 'decay_type':r['cfg']['decay_type'], 'A':r['cfg']['A'], 'G':r['cfg']['G'], 'tau':r['cfg']['tau'], 'alpha':r['alpha'], 'rmse':r['rmse'], 'mae':r['mae'], 'pred_std':r['pred_std'], 'actual_std':r['actual_std'], 'n_folds':r['n_folds'], 'stage':r['stage']} for r in results])
    summary = summary.sort_values(['rmse','mae','alpha','config_key'], kind='mergesort').reset_index(drop=True)
    best = sorted(results, key=lambda r:(r['rmse'], r['mae'], r['alpha'], cfg_key(r['cfg'])))[0]
    best_base = sorted([r for r in results if r['cfg']['A'] <= 0], key=lambda r:(r['rmse'], r['mae'], r['alpha']))[0]
    meta = {'tuning_elapsed_seconds': round(time.perf_counter()-b['tune_start'],4), 'model_fit_count': int(b['fit_count']), 'tuning_stop_reason': b.get('stop_reason'), 'n_eval_rows': int(len(summary)), 'n_unique_elo_configs': int(summary['config_key'].nunique()), 'n_stage1': int((summary['stage']=='stage1').sum()), 'n_stage2': int((summary['stage']=='stage2').sum())}
    return best, best_base, results, summary, elo_cache, meta

def fit_final(train_df, pred_df, rankings_template, best_res, season_ctx, home_adv):
    elo_feat, final_ratings = run_elo_train(train_df, season_ctx, home_adv, best_res['cfg'])
    derby_elo = run_elo_derby(pred_df, final_ratings)
    team_ids = sorted(set(train_df['HomeID'].astype(int)) | set(train_df['AwayID'].astype(int)))
    massey = fit_massey_ridge(train_df, team_ids=team_ids, alpha=30.0)
    tr_m = apply_massey_features(train_df, massey, 'HomeID', 'AwayID', neutral_site=False)[['massey_diff']]
    de_m = apply_massey_features(pred_df, massey, 'Team1_ID', 'Team2_ID', neutral_site=True)[['massey_diff']]

    Xtr = pd.concat([elo_feat[['elo_diff_pre']].reset_index(drop=True), tr_m.reset_index(drop=True), conf_dummies(train_df,'HomeConf','AwayConf').reset_index(drop=True)], axis=1)
    Xde = pd.concat([derby_elo[['elo_diff_pre']].reset_index(drop=True), de_m.reset_index(drop=True), conf_dummies(pred_df,'Team1_Conf','Team2_Conf').reset_index(drop=True)], axis=1)
    Xtr.columns = [str(c) for c in Xtr.columns]; Xde.columns = [str(c) for c in Xde.columns]
    Xde = Xde.reindex(columns=Xtr.columns, fill_value=0.0)
    y = train_df['HomeWinMargin'].to_numpy(float)
    mdl = ridge_pipe(best_res['alpha']); mdl.fit(Xtr.astype(float), y)
    raw = np.asarray(mdl.predict(Xde.astype(float)), float)
    qlo, qhi = float(np.quantile(y,0.005)), float(np.quantile(y,0.995))
    clipped = np.clip(raw, qlo, qhi)
    rounded = np.rint(clipped).astype(int)
    clip_applied = bool(np.any(np.abs(raw-clipped)>1e-12))

    pred_out = pred_df.copy(); pred_out['Team1_WinMargin'] = rounded.astype(int)
    rank_out = rankings_template.copy(); rank_out['TeamID'] = rank_out['TeamID'].astype(int)
    if 'Team' in rank_out.columns: rank_out['Team'] = rank_out['Team'].astype(str)
    rank_out['_elo'] = rank_out['TeamID'].map(final_ratings).fillna(INITIAL_ELO).astype(float)
    rank_out = rank_out.sort_values(['_elo','TeamID'], ascending=[False,True], kind='mergesort').reset_index(drop=True)
    rank_out['Rank'] = np.arange(1, len(rank_out)+1, dtype=int)
    rank_out = rank_out.drop(columns=['_elo'])
    cols = [c for c in ['TeamID','Team','Rank'] if c in rank_out.columns] + [c for c in rank_out.columns if c not in {'TeamID','Team','Rank'}]
    rank_out = rank_out[cols]

    return {'elo_feat':elo_feat,'final_ratings':final_ratings,'pred_out':pred_out,'rank_out':rank_out,'raw':raw,'clipped':clipped,'rounded':rounded,'clip_bounds':(qlo,qhi),'clip_applied':clip_applied}


def git_short_hash():
    try: return subprocess.check_output(['git','rev-parse','--short','HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return 'unknown'


def git_branch():
    try: return subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return 'unknown'


def draw_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.5,11)); fig.suptitle(title, fontsize=15, fontweight='bold', y=0.985)
    wrapped = []
    for line in lines:
        wrapped.extend([''] if line == '' else (textwrap.wrap(str(line), width=105) or ['']))
    fig.text(0.05,0.95,'\n'.join(wrapped), va='top', ha='left', fontsize=10, family='monospace')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def draw_table_page(pdf, title, df, float_cols=None):
    float_cols = float_cols or []
    show = df.copy()
    for c in float_cols:
        if c in show.columns: show[c] = show[c].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else '')
    fig, ax = plt.subplots(figsize=(11,8.5)); ax.axis('off'); ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc='center'); tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0,1.2)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def generate_report(pdf_path, best_res, best_base, summary, tuning_meta, budget, final_fit, validation):
    if best_res['cfg']['A'] > 0:
        best_k = best_res
    else:
        krows = summary[summary['A']>0].sort_values(['rmse','mae'], kind='mergesort') if len(summary) else pd.DataFrame()
        best_k = None if len(krows)==0 else {'cfg': {'decay_type':str(krows.iloc[0]['decay_type']), 'A':float(krows.iloc[0]['A']), 'G':int(krows.iloc[0]['G']), 'tau':int(krows.iloc[0]['tau'])}, 'alpha':float(krows.iloc[0]['alpha']), 'rmse':float(krows.iloc[0]['rmse']), 'mae':float(krows.iloc[0]['mae']), 'pred_std':float(krows.iloc[0]['pred_std']), 'actual_std':float(krows.iloc[0]['actual_std'])}
    results_rows = [{'Variant':'Baseline (A=0)','DecayType':best_base['cfg']['decay_type'],'A':best_base['cfg']['A'],'G':best_base['cfg']['G'],'tau':best_base['cfg']['tau'],'alpha':best_base['alpha'],'RMSE':best_base['rmse'],'MAE':best_base['mae'],'OOF_Pred_STD':best_base['pred_std'],'OOF_Actual_STD':best_base['actual_std']}]
    if best_k is not None:
        results_rows.append({'Variant':'Best K-decay (A>0)','DecayType':best_k['cfg']['decay_type'],'A':best_k['cfg']['A'],'G':best_k['cfg']['G'],'tau':best_k['cfg']['tau'],'alpha':best_k['alpha'],'RMSE':best_k['rmse'],'MAE':best_k['mae'],'OOF_Pred_STD':best_k['pred_std'],'OOF_Actual_STD':best_k['actual_std']})
    res_tbl = pd.DataFrame(results_rows)

    oof = best_res['oof'].copy(); oof['resid'] = oof['y_true'] - oof['y_pred']
    lines = [
        'Baseline reference: commit 9ea582d (known RMSE ≈ 286).',
        'Why simplicity: recent complex approach regressed to RMSE ≈ 354, so this run uses simple Ridge + stable matchup features for better generalization.',
        '',
        'HSAC K-decay Elo:',
        'E_home = 1 / (1 + 10^((R_away - (R_home + HOME_ADV))/400))',
        'Update = K_base * K_mult(g) * MOV_mult * (S - E_home)',
        'Linear K_mult(g) = 1 + A * max(0, 1 - g/G)',
        'Exponential K_mult(g) = 1 + A * exp(-g/tau)',
        'Season breaks detected by Date gap > 60 days; single-season => season_id=0.',
        '',
        f'Grid: alpha={ALPHA_GRID}; A={A_GRID}; G={G_GRID}; tau={TAU_GRID}; decay_type=[linear, exponential]',
        f"Budgets: MAX_TOTAL_SECONDS={budget['max_total']}, MAX_TUNING_SECONDS={budget['max_tune']}, MAX_MODEL_FITS={budget['max_fits']}",
        f"Budget usage: tuning_elapsed={tuning_meta['tuning_elapsed_seconds']}s, fit_count={tuning_meta['model_fit_count']}, stop_reason={tuning_meta['tuning_stop_reason']}",
        'Time-aware CV: expanding-window folds on Train.csv sorted by Date then GameID.',
        f"Chosen settings: decay_type={best_res['cfg']['decay_type']}, A={best_res['cfg']['A']}, G={best_res['cfg']['G']}, tau={best_res['cfg']['tau']}, HOME_ADV={HOME_ADV}, ridge_alpha={best_res['alpha']}",
        f"Chosen CV RMSE/MAE = {best_res['rmse']:.3f}/{best_res['mae']:.3f}; baseline A=0 RMSE/MAE = {best_base['rmse']:.3f}/{best_base['mae']:.3f}",
        ('K-decay helped (A>0 selected).' if best_res['cfg']['A'] > 0 else 'K-decay did not help; A=0 retained.'),
    ]

    with PdfPages(pdf_path) as pdf:
        draw_text_page(pdf, 'Final Report: Simple K-decay Elo + Ridge', lines)
        draw_table_page(pdf, 'Results Table: Baseline vs Best K-decay', res_tbl, float_cols=['A','alpha','RMSE','MAE','OOF_Pred_STD','OOF_Actual_STD'])
        draw_table_page(pdf, 'Top CV Configurations (Budgeted Search)', summary.head(15)[['config_key','decay_type','A','G','tau','alpha','rmse','mae','pred_std','actual_std','stage']], float_cols=['A','alpha','rmse','mae','pred_std','actual_std'])

        fig, axes = plt.subplots(2,2, figsize=(11,8.5))
        ax=axes[0,0]; ax.scatter(oof['y_pred'], oof['y_true'], s=12, alpha=0.6); lo=min(oof['y_true'].min(), oof['y_pred'].min()); hi=max(oof['y_true'].max(), oof['y_pred'].max()); ax.plot([lo,hi],[lo,hi], color='black', lw=1); ax.set_title('OOF predicted vs actual')
        ax=axes[0,1]; ax.hist(oof['resid'], bins=30, color='#4C78A8', alpha=0.85); ax.axvline(0,color='black',lw=1); ax.set_title('Residual histogram')
        ax=axes[1,0]; ax.scatter(oof['y_pred'], oof['resid'], s=12, alpha=0.6, color='#F58518'); ax.axhline(0,color='black',lw=1); ax.set_title('Residual vs fitted')
        ax=axes[1,1]; ax.hist(final_fit['rounded'], bins=20, color='#54A24B', alpha=0.85); ax.set_title('Distribution of derby predictions')
        fig.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        fig, axes = plt.subplots(2,1, figsize=(11,8.5), gridspec_kw={'height_ratios':[1.0,1.2]})
        ax=axes[0]; vals=[best_res['pred_std'], best_res['actual_std']]; labels=['OOF pred std','OOF actual std']; ax.bar(labels, vals, color=['#4C78A8','#E45756']); ax.set_title('Dispersion comparison (OOF)'); [ax.text(i,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9) for i,v in enumerate(vals)]
        ax=axes[1]; ax.axis('off'); ax.text(0.02,0.98,'\n'.join([
            'Artifact validation proofs:',
            f"predictions.csv rows={validation['pred_rows']}, missing={validation['pred_missing']}, numeric={validation['pred_numeric']}",
            f"rankings.xlsx rows={validation['rank_rows']}, rank_missing={validation['rank_missing']}, rank_set_valid={validation['rank_set_valid']}",
            f"final_report.pdf exists={validation['pdf_exists']}, size_bytes={validation['pdf_size']}",
            f"Derby clipping (train 0.5/99.5 pct): {'applied' if final_fit['clip_applied'] else 'not applied'}; bounds=({final_fit['clip_bounds'][0]:.2f}, {final_fit['clip_bounds'][1]:.2f})",
            '',
            'Tuning meta:', json.dumps(tuning_meta, indent=2)
        ]), va='top', ha='left', family='monospace', fontsize=10)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def validate_and_print(pred_path, rank_path, pdf_path):
    p = pd.read_csv(pred_path); r = pd.read_excel(rank_path)
    pred_rows = int(len(p)); pred_missing = int(p['Team1_WinMargin'].isna().sum()); pred_numeric = bool(pd.api.types.is_numeric_dtype(p['Team1_WinMargin']))
    rank_rows = int(len(r)); rank_missing = int(r['Rank'].isna().sum()); rank_set_valid = bool(set(pd.to_numeric(r['Rank'], errors='coerce').dropna().astype(int).tolist()) == set(range(1, rank_rows+1)))
    pdf_size = int(Path(pdf_path).stat().st_size)
    print('\n=== HARD STOP VALIDATIONS ===')
    print(f'predictions.csv exists=True rows={pred_rows} Team1_WinMargin numeric={pred_numeric} no_missing={pred_missing==0}')
    print(f'rankings.xlsx exists=True rows={rank_rows} Rank integer set {{1..{rank_rows}}} valid={rank_set_valid}')
    print(f'final_report.pdf exists=True size > 0 => {pdf_size>0} (size={pdf_size})')
    print('\nHead(10) predictions:'); print(p.head(10).to_string(index=False))
    print('\nHead(10) rankings sorted by Rank:'); print(r.sort_values('Rank', kind='mergesort').head(10).to_string(index=False))
    if pred_rows != 75 or pred_missing != 0 or not pred_numeric: raise ValueError('predictions.csv validation failed')
    if rank_rows != 165 or rank_missing != 0 or not rank_set_valid: raise ValueError('rankings.xlsx validation failed')
    if pdf_size <= 0: raise ValueError('final_report.pdf validation failed')
    return {'pred_rows':pred_rows,'pred_missing':pred_missing,'pred_numeric':pred_numeric,'rank_rows':rank_rows,'rank_missing':rank_missing,'rank_set_valid':rank_set_valid,'pdf_exists':True,'pdf_size':pdf_size}


def write_run_report(path, budget, tuning_meta, best_res, best_base, summary, validation, final_fit):
    krows = summary[summary['A']>0].sort_values(['rmse','mae'], kind='mergesort') if len(summary) else pd.DataFrame()
    txt = [
        '# run_report','',f'- git commit hash: `{git_short_hash()}`',f'- git branch: `{git_branch()}`','- baseline reference commit: `9ea582d`','',
        '## budgets used',f"- MAX_TOTAL_SECONDS: {budget['max_total']}",f"- MAX_TUNING_SECONDS: {budget['max_tune']}",f"- MAX_MODEL_FITS: {budget['max_fits']}",f"- tuning_elapsed_seconds: {tuning_meta['tuning_elapsed_seconds']}",f"- total_elapsed_seconds_at_write: {round(time.perf_counter()-budget['start'],3)}",f"- model_fit_count: {budget['fit_count']}",f"- tuning_stop_reason: {budget.get('stop_reason')}",'',
        '## chosen params',f"- decay_type: {best_res['cfg']['decay_type']}",f"- A: {best_res['cfg']['A']}",f"- G: {best_res['cfg']['G']}",f"- tau: {best_res['cfg']['tau']}",f"- HOME_ADV: {HOME_ADV}",f"- ridge_alpha: {best_res['alpha']}",'',
        '## cv metrics',f"- chosen_rmse: {best_res['rmse']:.6f}",f"- chosen_mae: {best_res['mae']:.6f}",f"- chosen_oof_pred_std: {best_res['pred_std']:.6f}",f"- chosen_oof_actual_std: {best_res['actual_std']:.6f}",f"- best_baseline_rmse (A=0): {best_base['rmse']:.6f}",f"- best_baseline_mae (A=0): {best_base['mae']:.6f}",f"- best_kdecay_rmse (A>0): {None if len(krows)==0 else float(krows.iloc[0]['rmse'])}",f"- best_kdecay_mae (A>0): {None if len(krows)==0 else float(krows.iloc[0]['mae'])}",'',
        '## output validations',f"- predictions.csv rows: {validation['pred_rows']}",f"- predictions.csv Team1_WinMargin missing: {validation['pred_missing']}",f"- predictions.csv Team1_WinMargin numeric: {validation['pred_numeric']}",f"- rankings.xlsx rows: {validation['rank_rows']}",f"- rankings.xlsx Rank missing: {validation['rank_missing']}",f"- rankings.xlsx rank_set_valid: {validation['rank_set_valid']}",f"- final_report.pdf exists: {validation['pdf_exists']}",f"- final_report.pdf size_bytes: {validation['pdf_size']}",f"- derby clipping applied: {final_fit['clip_applied']}",f"- derby clip bounds (train 0.5/99.5 pct): {final_fit['clip_bounds']}",'',
        '## tuning meta','```json',json.dumps(tuning_meta, indent=2),'```','','## top cv rows (first 10)','```',summary.head(10).to_string(index=False),'```'
    ]
    Path(path).write_text('\n'.join(txt), encoding='utf-8')

def main():
    seed_all(SEED)
    budget = {
        'start': time.perf_counter(),
        'max_total': env_float('ALGOSPORTS_MAX_TOTAL_SECONDS', 360.0),
        'max_tune': env_float('ALGOSPORTS_MAX_TUNING_SECONDS', 120.0),
        'max_fits': env_int('ALGOSPORTS_MAX_MODEL_FITS', 200),
        'fit_count': 0,
        'tune_start': None,
        'stop_reason': None,
    }
    print('=== AlgoSports Simple K-decay Elo + Ridge ===')
    print(f"Thread caps: OMP={os.getenv('OMP_NUM_THREADS')} MKL={os.getenv('MKL_NUM_THREADS')} OPENBLAS={os.getenv('OPENBLAS_NUM_THREADS')} NUMEXPR={os.getenv('NUMEXPR_NUM_THREADS')}")
    print(f"Budgets: MAX_TOTAL_SECONDS={budget['max_total']} MAX_TUNING_SECONDS={budget['max_tune']} MAX_MODEL_FITS={budget['max_fits']}")

    train_path, pred_template_path, rankings_template_path = ensure_root_inputs(ROOT)
    print(f"Root inputs confirmed: {train_path.name}, {pred_template_path.name}, {rankings_template_path.name}")

    train_df = parse_and_sort_train(pd.read_csv(train_path))
    pred_df = parse_predictions(pd.read_csv(pred_template_path))
    rankings_template = pd.read_excel(rankings_template_path)

    season_ctx = detect_seasons(train_df['Date'])
    n_seasons = len(set(season_ctx[0].tolist())) if len(season_ctx[0]) else 0
    print(f'Season detection: {n_seasons} season(s) using gap > 60 days rule')

    try:
        folds = make_expanding_time_folds(train_df, n_folds=5)
    except Exception as e:
        print(f'5-fold expanding CV unavailable ({e}); falling back to 4 folds')
        folds = make_expanding_time_folds(train_df, n_folds=4)
    print(f'Using {len(folds)} expanding-window folds')

    print(f'HOME_ADV fixed at {HOME_ADV} for simplicity')
    fold_caches = build_fold_caches(train_df, folds)
    print('Fold caches built (Massey + conference one-hot)')

    best_res, best_base, _results, summary, _elo_cache, tuning_meta = tune_budgeted(train_df, fold_caches, season_ctx, HOME_ADV, budget)
    print(f"Chosen config: {cfg_label(best_res['cfg'])}, alpha={best_res['alpha']}")
    print(f"Chosen CV RMSE={best_res['rmse']:.3f}, MAE={best_res['mae']:.3f}")
    print(f"Best baseline (A=0) RMSE={best_base['rmse']:.3f}, MAE={best_base['mae']:.3f}")
    if best_res['cfg']['A'] <= 0:
        print('K-decay did not improve over A=0 under budgeted CV')

    final_fit = fit_final(train_df, pred_df, rankings_template, best_res, season_ctx, HOME_ADV)

    pred_out_path = ROOT / 'predictions.csv'
    rank_out_path = ROOT / 'rankings.xlsx'
    pdf_out_path = ROOT / 'final_report.pdf'
    run_report_path = ROOT / 'run_report.md'
    final_fit['pred_out'].to_csv(pred_out_path, index=False)
    final_fit['rank_out'].to_excel(rank_out_path, index=False)

    pre_pdf_validation = {
        'pred_rows': int(len(final_fit['pred_out'])),
        'pred_missing': int(final_fit['pred_out']['Team1_WinMargin'].isna().sum()),
        'pred_numeric': bool(pd.api.types.is_numeric_dtype(final_fit['pred_out']['Team1_WinMargin'])),
        'rank_rows': int(len(final_fit['rank_out'])),
        'rank_missing': int(final_fit['rank_out']['Rank'].isna().sum()),
        'rank_set_valid': bool(set(final_fit['rank_out']['Rank'].astype(int).tolist()) == set(range(1, len(final_fit['rank_out'])+1))),
        'pdf_exists': False,
        'pdf_size': 0,
    }
    generate_report(pdf_out_path, best_res, best_base, summary, tuning_meta, budget, final_fit, pre_pdf_validation)

    validation = validate_and_print(pred_out_path, rank_out_path, pdf_out_path)
    # Regenerate report once so embedded artifact validation proofs include actual PDF existence/size.
    generate_report(pdf_out_path, best_res, best_base, summary, tuning_meta, budget, final_fit, validation)
    validation = validate_and_print(pred_out_path, rank_out_path, pdf_out_path)
    write_run_report(run_report_path, budget, tuning_meta, best_res, best_base, summary, validation, final_fit)
    print(f"\nrun_report.md written: {run_report_path.name}")
    total_elapsed = time.perf_counter() - budget['start']
    print(f'Total elapsed seconds: {total_elapsed:.2f}')
    if total_elapsed > budget['max_total']:
        print('WARNING: total runtime exceeded MAX_TOTAL_SECONDS, but artifacts were generated as required')


if __name__ == '__main__':
    main()
