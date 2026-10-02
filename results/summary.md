## Outcome per scenario (✓ = outcome matches the expected one)

| Scenario | Expected | react | plan_execute | hybrid |
|---|---|---|---|---|
| valid | done | ✓ done (-) | ✓ done (-) | ✓ done (-) |
| no_valid | handoff | ✓ handoff (gave_up) | ✓ handoff (gave_up) | ✓ handoff (gave_up) |
| trap | done | ✓ done (-) | ✓ done (-) | ✓ done (-) |
| transient_error | done | ✓ done (-) | ✗ handoff (step_failed) | ✓ done (-) |
| sold_out | done | ✓ done (-) | ✗ handoff (step_failed) | ✓ done (-) |
| needs_approval | handoff | ✓ handoff (needs_approval) | ✓ handoff (needs_approval) | ✓ handoff (needs_approval) |

## Per pattern

| Pattern | Runs | Correct | Done when expected | Handoff when expected | Blocked calls | Model calls (avg) | Tokens (avg) | Seconds (avg) | Infra errors |
|---|---|---|---|---|---|---|---|---|---|
| react | 6 | 6/6 | 4/4 | 2/2 | 1 | 5.3 | 5766 | 9.5 | 0 |
| plan_execute | 6 | 4/6 | 2/4 | 2/2 | 1 | 1.0 | 640 | 4.4 | 0 |
| hybrid | 6 | 6/6 | 4/4 | 2/2 | 1 | 1.3 | 908 | 2.1 | 0 |
