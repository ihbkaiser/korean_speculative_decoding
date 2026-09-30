# Offline detector validation

This is a CPU-only feasibility audit of saved baseline traces; it is not a runtime or speed claim.
For proposals before the first baseline rejection (and all proposals in fully accepted rounds), exact H3 target-token labels are reused because the proposed token ID equals the saved target continuation ID. The first rejected candidate, when it is an internal slot, is explicitly reprojected from its saved draft ID with the H2 tokenizer-span routine and H3 POS transition function. Opportunities are counted only through the first-rejection slot; later proposals cannot have a fully accepted safe prefix and are not reachable guard opportunities.

## Opportunity counts

| workload  | pair | policy                   | candidate_blocks | prompts | mean_guard_slot    | mean_positions_after | reachable_share |
| --------- | ---- | ------------------------ | ---------------- | ------- | ------------------ | -------------------- | --------------- |
| FLORES    | P1   | ALL_MORPH_BOUNDARY_GUARD | 1839             | 802     | 1.7422512234910277 | 2.1963023382272975   | 1.0             |
| FLORES    | P1   | NP_BOUNDARY_GUARD        | 214              | 164     | 1.6261682242990654 | 2.369158878504673    | 1.0             |
| FLORES    | P2   | ALL_MORPH_BOUNDARY_GUARD | 1776             | 785     | 1.7961711711711712 | 2.1458333333333335   | 1.0             |
| FLORES    | P2   | NP_BOUNDARY_GUARD        | 227              | 191     | 1.748898678414097  | 2.251101321585903    | 1.0             |
| FLORES    | P3   | ALL_MORPH_BOUNDARY_GUARD | 1698             | 768     | 1.7444051825677267 | 2.177856301531213    | 1.0             |
| FLORES    | P3   | NP_BOUNDARY_GUARD        | 175              | 154     | 1.6571428571428573 | 2.342857142857143    | 1.0             |
| WIKIPEDIA | P1   | ALL_MORPH_BOUNDARY_GUARD | 2293             | 613     | 1.8918447448757088 | 2.094635848233755    | 1.0             |
| WIKIPEDIA | P1   | NP_BOUNDARY_GUARD        | 171              | 92      | 1.8128654970760234 | 2.181286549707602    | 1.0             |
| WIKIPEDIA | P2   | ALL_MORPH_BOUNDARY_GUARD | 2386             | 649     | 1.9149203688181056 | 2.0729253981559093   | 1.0             |
| WIKIPEDIA | P2   | NP_BOUNDARY_GUARD        | 192              | 116     | 1.8229166666666667 | 2.1666666666666665   | 1.0             |
| WIKIPEDIA | P3   | ALL_MORPH_BOUNDARY_GUARD | 2444             | 645     | 1.8461538461538463 | 2.1382978723404253   | 1.0             |
| WIKIPEDIA | P3   | NP_BOUNDARY_GUARD        | 174              | 111     | 1.764367816091954  | 2.2298850574712645   | 1.0             |

## Frozen entropy thresholds

| pair | threshold         | matched_blocks | eligible_blocks_before_thinning | eligible_slot_histogram        | target_np_slot_histogram    | slot_thinning_probabilities                                                  | extra_eligible_blocks | score                           | target_np_activation_blocks | calibration_source                                                                                      | method                                                                                                                                                   |
| ---- | ----------------- | -------------- | ------------------------------- | ------------------------------ | --------------------------- | ---------------------------------------------------------------------------- | --------------------- | ------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1   | 5.291721978187556 | 171            | 500                             | {'3': 96, '1': 275, '2': 129}  | {'1': 72, '2': 59, '3': 40} | {'1': 0.26181818181818184, '2': 0.4573643410852713, '3': 0.4166666666666667} | 329                   | [-171, 329, -5.291721978187556] | 171                         | WIKIPEDIA only; saved teacher-forced draft entropy at positions with accepted preceding proposal prefix | 201 empirical entropy quantiles; choose the cutoff maximizing slotwise-supported NP activations and freeze deterministic slotwise thinning probabilities |
| P2   | 5.098802773952483 | 192            | 504                             | {'1': 277, '3': 102, '2': 125} | {'2': 52, '1': 87, '3': 53} | {'2': 0.416, '1': 0.3140794223826715, '3': 0.5196078431372549}               | 312                   | [-192, 312, -5.098802773952483] | 192                         | WIKIPEDIA only; saved teacher-forced draft entropy at positions with accepted preceding proposal prefix | 201 empirical entropy quantiles; choose the cutoff maximizing slotwise-supported NP activations and freeze deterministic slotwise thinning probabilities |
| P3   | 5.329163768291475 | 174            | 511                             | {'1': 299, '2': 120, '3': 92}  | {'1': 75, '3': 34, '2': 65} | {'1': 0.2508361204013378, '3': 0.3695652173913043, '2': 0.5416666666666666}  | 337                   | [-174, 337, -5.329163768291475] | 174                         | WIKIPEDIA only; saved teacher-forced draft entropy at positions with accepted preceding proposal prefix | 201 empirical entropy quantiles; choose the cutoff maximizing slotwise-supported NP activations and freeze deterministic slotwise thinning probabilities |

## Alignment caveat

The offline opportunity table conservatively excludes any candidate whose H3 token record is missing or whose exact rejected-token projection is ambiguous. The online controller uses the full saved H2/H3 projection and disables guarding for a block whenever the full candidate sequence does not round-trip exactly.
