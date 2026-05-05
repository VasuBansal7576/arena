# Research Notes

## squared_error_no_cap_1m_500_lr04

- Hypothesis: A slower 500-iteration squared-error learner should reduce underfit without changing the inference contract.
- Risk: May overfit noisy late-2023 patterns if pushed further.
- Best before: 250.291827
- Best after: 250.291827
- Decision: PROMOTED

## route_class_specialists_1m_500_lr04

- Hypothesis: Route-class specialists can learn different airport, Manhattan, and outer-borough residual patterns.
- Risk: Specialists can overfit the Dev calibration window and add inference complexity.
- Full Dev result: 250.106341 vs 250.291827 baseline.
- Time holdout: 259.178048 vs 259.104298 baseline.
- Decision: REJECTED for submission because the later time-holdout got worse.

## route_class_specialists_pruned_1m_500_lr04

- Hypothesis: Keep only route-class specialists that pass the later time-holdout gate.
- Risk: Uses more inference paths, so the gain must survive both full Dev and holdout.
- Full Dev result: 250.265852 vs 250.291827 baseline.
- Time holdout: 259.071979 vs 259.104298 baseline.
- Decision: PROMOTED. The Manhattan-to/from-outer specialist kept a 0.2 blend; airport and Manhattan-internal specialists were pruned to 0.0.

## target_encoding_1m_500_lr04

- Hypothesis: Smoothed pair and pair-hour mean target encodings can expose residual signal not captured by median priors.
- Risk: Mean target encodings can chase noisy/high-variance routes and hurt temporal robustness.
- Full Dev result: 250.268991 vs 250.265852 baseline.
- Time holdout: 259.584251 vs 259.071979 baseline.
- Decision: REJECTED. It did not beat full Dev and significantly worsened the later holdout.

## variance_route_class_pruned_1m_500_lr04

- Hypothesis: Historical pair and pair-hour duration variance can tell the model which routes are unstable and improve calibration on noisy segments.
- Risk: Variance may mostly identify irreducible noise and pull the model away from the robust median-like target.
- Full Dev result: 250.832541 vs 250.265852 baseline.
- Time holdout: 259.216326 vs 259.071979 baseline.
- Decision: REJECTED. Route-class specialists improved within the experiment, but the added variance features worsened both full Dev and the late-December holdout versus the current best.

## residual_calibration_route_pruned_1m_500_lr04

- Hypothesis: Median residual correction tables should directly target MAE after model inference.
- Risk: If the model is already median-calibrated, additive residual tables may add noise or select zero weight.
- Full Dev result: 250.265852 vs 250.265852 baseline.
- Time holdout: 259.071979 vs 259.071979 baseline.
- Decision: REJECTED. Every residual correction alpha was selected as 0.0.

## affine_calibration_route_pruned_1m_500_lr04

- Hypothesis: MAE can improve more from segment-specific scale/offset calibration than from more generic features.
- Risk: This is the most metric-aware layer and can overfit Dev, so rules are fitted before the cutoff and kept only when the later time-holdout also improves.
- Full Dev result: 245.502003 vs 250.265852 baseline.
- Time holdout: 253.722303 vs 259.071979 baseline.
- Decision: PROMOTED. The final model keeps 43 route/hour/day/dropoff affine calibration rules.

## fine_affine_calibration_route_pruned_1m_500_lr04

- Hypothesis: Higher-resolution interaction calibration can recover remaining MAE in route-hour, day-hour, airport-hour, dropoff-hour, and route-dropoff pockets.
- Risk: This is more Dev-sensitive than the coarse affine layer, so each rule must still pass the later time-holdout gate.
- Full Dev result: 244.131899 vs 245.502003 baseline.
- Time holdout: 252.742993 vs 253.722303 baseline.
- Decision: PROMOTED. The final model keeps 116 affine calibration rules.
