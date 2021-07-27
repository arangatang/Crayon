from crayon import verify

for i in ["abs_error", "MASE", "MAPE", "RMSE", "MSIS"]:
    print("verifying bugged to fixed on", i)
    verify("deepar_fixed_100", "deepar_bugged_100", i)

    print("verifying bugged to pre-bug on", i)
    verify("deepar_pre_bug_100", "deepar_bugged_100", i)

    print("verifying bugged to bugged on", i)
    verify("deepar_bugged_100", "deepar_bugged_100", i)

    print("verifying pre-bug to pre-bug on", i)
    verify("deepar_pre_bug_100", "deepar_pre_bug_100", i)

    print("verifying pre-bug to fixed on", i)
    verify("deepar_pre_bug_100", "deepar_fixed_100", i)

    print("verifying fixed to fixed on", i)
    verify("deepar_fixed_100", "deepar_fixed_100", i)
    print("")