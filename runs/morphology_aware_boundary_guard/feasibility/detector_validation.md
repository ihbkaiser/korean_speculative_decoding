# Offline detector validation

This is a CPU-only feasibility audit of saved baseline traces; it is not a runtime or speed claim.
For proposals before the first baseline rejection (and all proposals in fully accepted rounds), exact H3 target-token labels are reused because the proposed token ID equals the saved target continuation ID. At and after a rejection, the entire saved draft proposal block is reprojected from the committed target prefix through the H2 tokenizer-span routine and H3 POS transition function. This records both reachable opportunities and candidates following a rejection; the latter are marked unreachable for that baseline block.

## Opportunity counts

| workload  | pair | policy                   | candidate_blocks | prompts | mean_guard_slot    | mean_positions_after | reachable_share    |
| --------- | ---- | ------------------------ | ---------------- | ------- | ------------------ | -------------------- | ------------------ |
| FLORES    | P1   | ALL_MORPH_BOUNDARY_GUARD | 2394             | 847     | 1.9540517961570594 | 1.9995822890559733   | 0.7339181286549707 |
| FLORES    | P1   | NP_BOUNDARY_GUARD        | 351              | 252     | 1.8518518518518519 | 2.1452991452991452   | 0.7065527065527065 |
| FLORES    | P2   | ALL_MORPH_BOUNDARY_GUARD | 2065             | 812     | 1.9394673123486683 | 2.012590799031477    | 0.8174334140435835 |
| FLORES    | P2   | NP_BOUNDARY_GUARD        | 339              | 261     | 1.8820058997050146 | 2.117994100294985    | 0.7522123893805309 |
| FLORES    | P3   | ALL_MORPH_BOUNDARY_GUARD | 2158             | 806     | 1.9624652455977758 | 1.9772937905468027   | 0.741890639481001  |
| FLORES    | P3   | NP_BOUNDARY_GUARD        | 269              | 216     | 1.858736059479554  | 2.141263940520446    | 0.7360594795539034 |
| WIKIPEDIA | P1   | ALL_MORPH_BOUNDARY_GUARD | 2616             | 644     | 2.000382262996942  | 1.9877675840978593   | 0.841743119266055  |
| WIKIPEDIA | P1   | NP_BOUNDARY_GUARD        | 217              | 117     | 1.9493087557603688 | 2.046082949308756    | 0.8110599078341014 |
| WIKIPEDIA | P2   | ALL_MORPH_BOUNDARY_GUARD | 2693             | 660     | 2.019680653546231  | 1.9695506869662087   | 0.8529520980319346 |
| WIKIPEDIA | P2   | NP_BOUNDARY_GUARD        | 246              | 144     | 1.9593495934959348 | 2.032520325203252    | 0.7804878048780488 |
| WIKIPEDIA | P3   | ALL_MORPH_BOUNDARY_GUARD | 2859             | 670     | 1.9972018188177685 | 1.9891570479188527   | 0.8104232249038125 |
| WIKIPEDIA | P3   | NP_BOUNDARY_GUARD        | 236              | 148     | 1.9957627118644068 | 2.0                  | 0.7372881355932204 |

## Frozen entropy thresholds

| pair | threshold         | matched_blocks | eligible_blocks_before_thinning | eligible_slot_histogram        | target_np_slot_histogram    | slot_thinning_probabilities                                                  | extra_eligible_blocks | score                           | target_np_activation_blocks | calibration_source                                                                                      | method                                                                                                                                                   |
| ---- | ----------------- | -------------- | ------------------------------- | ------------------------------ | --------------------------- | ---------------------------------------------------------------------------- | --------------------- | ------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1   | 5.291721978187556 | 217            | 500                             | {'3': 96, '1': 275, '2': 129}  | {'1': 74, '2': 80, '3': 63} | {'1': 0.2690909090909091, '2': 0.6201550387596899, '3': 0.65625}             | 283                   | [-217, 283, -5.291721978187556] | 217                         | WIKIPEDIA only; saved teacher-forced draft entropy at positions with accepted preceding proposal prefix | 201 empirical entropy quantiles; choose the cutoff maximizing slotwise-supported NP activations and freeze deterministic slotwise thinning probabilities |
| P2   | 5.098802773952483 | 246            | 504                             | {'1': 277, '3': 102, '2': 125} | {'2': 78, '3': 79, '1': 89} | {'2': 0.624, '3': 0.7745098039215687, '1': 0.3212996389891697}               | 258                   | [-246, 258, -5.098802773952483] | 246                         | WIKIPEDIA only; saved teacher-forced draft entropy at positions with accepted preceding proposal prefix | 201 empirical entropy quantiles; choose the cutoff maximizing slotwise-supported NP activations and freeze deterministic slotwise thinning probabilities |
| P3   | 5.329163768291475 | 236            | 511                             | {'1': 299, '2': 120, '3': 92}  | {'2': 83, '3': 76, '1': 77} | {'2': 0.6916666666666667, '3': 0.8260869565217391, '1': 0.25752508361204013} | 275                   | [-236, 275, -5.329163768291475] | 236                         | WIKIPEDIA only; saved teacher-forced draft entropy at positions with accepted preceding proposal prefix | 201 empirical entropy quantiles; choose the cutoff maximizing slotwise-supported NP activations and freeze deterministic slotwise thinning probabilities |

## Alignment caveat

The offline opportunity table conservatively excludes any candidate whose H3 token record is missing or whose exact rejected-token projection is ambiguous. The online controller uses the full saved H2/H3 projection and disables guarding for a block whenever the full candidate sequence does not round-trip exactly.
