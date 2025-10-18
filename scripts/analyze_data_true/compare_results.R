# Comparison script for Ground Truth vs AI Implementation
library(did)
library(dplyr)

cat('\n========== COMPARISON OF GROUND TRUTH vs AI IMPLEMENTATION ==========\n\n')

# Dataset 1: df_design_cov (with covariates)
cat('### Dataset 1: df_design_cov (with covariates) ###\n')
df_cov <- readRDS('output/difference_in_differences_multiperiods/df_design_cov.rds')

# Ground truth approach (from difference_in_differences_multiperiods.Rmd line 907-917)
att_gt_cov_true <- att_gt(
  yname = 'y_it',
  tname = 'time',
  idname = 'id',
  gname = 'group_i',
  data = df_cov,
  xformla = ~1 + x1_i + x2_i
)
att_simple_true <- aggte(att_gt_cov_true, type = 'simple')

# AI approach (from analyze_did_multiperiods.Rmd line 201-210)
att_gt_cov_ai <- att_gt(
  yname = 'y_it',
  tname = 'time',
  idname = 'id',
  gname = 'group_i',
  xformla = ~x1_i + x2_i,
  data = df_cov,
  control_group = 'nevertreated',
  est_method = 'dr'
)
att_simple_ai <- aggte(att_gt_cov_ai, type = 'simple')

cat('Ground Truth Overall ATT: ', round(att_simple_true$overall.att, 6), '\n')
cat('AI Implementation Overall ATT: ', round(att_simple_ai$overall.att, 6), '\n')
cat('Difference: ', round(abs(att_simple_true$overall.att - att_simple_ai$overall.att), 8), '\n\n')

# Dataset 2: df_design_nocov (without covariates)
cat('### Dataset 2: df_design_nocov (without covariates) ###\n')
df_nocov <- readRDS('output/difference_in_differences_multiperiods/df_design_nocov.rds')

# Ground truth
att_gt_nocov_true <- att_gt(
  yname = 'y_it',
  tname = 'time',
  idname = 'id',
  gname = 'group_i',
  data = df_nocov
)
att_simple_nocov_true <- aggte(att_gt_nocov_true, type = 'simple')

# AI
att_gt_nocov_ai <- att_gt(
  yname = 'y_it',
  tname = 'time',
  idname = 'id',
  gname = 'group_i',
  xformla = ~1,
  data = df_nocov,
  control_group = 'nevertreated',
  est_method = 'dr'
)
att_simple_nocov_ai <- aggte(att_gt_nocov_ai, type = 'simple')

cat('Ground Truth Overall ATT: ', round(att_simple_nocov_true$overall.att, 6), '\n')
cat('AI Implementation Overall ATT: ', round(att_simple_nocov_ai$overall.att, 6), '\n')
cat('Difference: ', round(abs(att_simple_nocov_true$overall.att - att_simple_nocov_ai$overall.att), 8), '\n\n')

# Dataset 3: df_design_nyt (not-yet-treated)
cat('### Dataset 3: df_design_nyt (not-yet-treated control) ###\n')
df_nyt <- readRDS('output/difference_in_differences_multiperiods/df_design_nyt.rds')

# Ground truth (line 1011-1019, using notyettreated, NO covariates)
att_gt_nyt_true <- att_gt(
  yname = 'y_it',
  tname = 'time',
  idname = 'id',
  gname = 'group_i',
  data = df_nyt,
  control_group = 'notyettreated'
)
att_simple_nyt_true <- aggte(att_gt_nyt_true, type = 'simple')

# AI (line 557-566, using notyettreated, WITH covariates)
att_gt_nyt_ai <- att_gt(
  yname = 'y_it',
  tname = 'time',
  idname = 'id',
  gname = 'group_i',
  xformla = ~x1_i + x2_i,
  data = df_nyt,
  control_group = 'notyettreated',
  est_method = 'dr'
)
att_simple_nyt_ai <- aggte(att_gt_nyt_ai, type = 'simple')

cat('Ground Truth Overall ATT (no covariates): ', round(att_simple_nyt_true$overall.att, 6), '\n')
cat('AI Implementation Overall ATT (with covariates): ', round(att_simple_nyt_ai$overall.att, 6), '\n')
cat('Difference: ', round(abs(att_simple_nyt_true$overall.att - att_simple_nyt_ai$overall.att), 8), '\n')
cat('NOTE: Ground truth does NOT use covariates, AI does use covariates (x1_i + x2_i)\n\n')

cat('========== COMPARISON COMPLETE ==========\n')
