import pandas as pd, matplotlib.pyplot as plt

# 出力先のパス設定
    save_dir = os.path.join(
        parent_dir,
        "results",
        "E0",
        "M1"
        )
    os.makedirs(save_dir, exist_ok=True)

k1 = pd.read_csv("results/E0/M1/norm_report_mean.csv")
k5 = pd.read_csv("results/E0/M1/norm_report_mean.csv")

plt.figure(figsize=(9, 6))
plt.plot(k1["round"], k1["SI_conv"], marker="o", label="K=1 (centralized)")
plt.plot(k5["round"], k5["SI_conv"], marker="s", label="K=5 (FedAvg)")
plt.yscale("log")
plt.xlabel("Round"); plt.ylabel("SI_conv weight norm (log)")
plt.title("SI_conv norm, centralized vs FedAvg")
plt.grid(True, which="both"); plt.legend()
plt.savefig(os.path.join(save_dir, "SIconv_compare.png"))