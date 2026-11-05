import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BYTES = ["A3", "F1", "B0", "4C", "9E", "22"]
HEX = "".join(BYTES).lower()
N = 3
CW, CH = 0.62, 0.62  # cell width/height
X0 = 3.55
BLUE, BLUE_E, GREY, LINE = "#9ec5ea", "#2f6fae", "#f3f4f6", "#4b5563"

fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=300)
ax.set_xlim(0, X0 + 12 * CW + 2.6)
ax.set_ylim(-1.15, 6.25)
ax.axis("off")
F = "DejaVu Sans"


def cell(x, y, w, txt, fc="white", ec=LINE, lw=0.8, bold=False, fs=10):
    ax.add_patch(Rectangle((x, y), w, CH, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + CH / 2, txt, ha="center", va="center", fontsize=fs,
            family="monospace", weight="bold" if bold else "normal")


# Row 1: raw ciphertext bytes (each spans two hex cells)
y = 5.35
ax.text(0.05, y + CH / 2, "Ciphertext bytes $y'$\n(6 bytes, values 0-255)", va="center", fontsize=8.5, family=F)
for i, b in enumerate(BYTES):
    cell(X0 + 2 * i * CW, y, 2 * CW, b, fc=GREY, bold=True)
    ax.text(X0 + (2 * i + 1) * CW, y + CH + 0.13, f"$b_{i+1}$", ha="center", fontsize=8)

# Arrow + encoding label
ax.annotate("", xy=(X0 + 6 * CW, 4.62), xytext=(X0 + 6 * CW, 5.3),
            arrowprops=dict(arrowstyle="-|>", lw=1.0, color=LINE))
ax.text(X0 + 6 * CW + 0.15, 4.93, r"lossless hex encoding $H(\cdot)$ (each byte $\rightarrow$ 2 characters)",
        fontsize=8, va="center", family=F)

# Row 2: hex-character string
y = 3.95
ax.text(0.05, y + CH / 2, "Hex-character string $h$\n(12 characters, 16 symbols)", va="center", fontsize=8.5, family=F)
for k, ch in enumerate(HEX):
    cell(X0 + k * CW, y, CW, ch, bold=True)
    ax.text(X0 + (k + 0.5) * CW, y - 0.17, f"{k+1}", ha="center", fontsize=6.5, color="#6b7280")
for i in range(7):  # byte boundaries
    ax.plot([X0 + 2 * i * CW] * 2, [y - 0.05, y + CH + 0.05], color="black", lw=2.0)

# Rows 3..6: sliding windows of N=3 hex characters
spans = ["$b_1$ + high nibble of $b_2$", "low nibble of $b_1$ + $b_2$",
         "$b_2$ + high nibble of $b_3$", "low nibble of $b_2$ + $b_3$"]
ax.text(0.05, 2.35, "Sliding window\n($N = 3$ hex characters,\nstride 1 character)", va="center", fontsize=8.5, family=F)
for r in range(4):
    y = 3.05 - r * 0.78
    for k, ch in enumerate(HEX):
        on = r <= k < r + N
        cell(X0 + k * CW, y, CW, ch, fc=BLUE if on else "white",
             ec=BLUE_E if on else "#c7cbd1", lw=1.2 if on else 0.6, bold=on, fs=9)
    for i in range(7):
        ax.plot([X0 + 2 * i * CW] * 2, [y, y + CH], color="#9ca3af", lw=1.2)
    ax.add_patch(Rectangle((X0 + r * CW, y), N * CW, CH, fill=False, ec=BLUE_E, lw=1.8))
    g = HEX[r:r + N]
    ax.text(X0 + 12 * CW + 0.2, y + CH / 2 + 0.1, f"$g_3({r+1})$ = \"{g}\"", fontsize=8.5, va="center", family=F)
    ax.text(X0 + 12 * CW + 0.2, y + CH / 2 - 0.16, spans[r], fontsize=6.8, va="center", color="#374151", family=F)

ax.text(X0 + 6 * CW, -0.08, r"$\vdots$  continues to $g_3(10)$ = 'e22';  $2L' - N + 1 = 10$ windows for $L' = 6$ bytes",
        ha="center", fontsize=8, family=F)
ax.text(0.05, -0.75,
        "Each order-3 N-gram spans 1.5 original bytes and may straddle a byte boundary (thick lines). "
        "Vocabulary per order: $16^N$\n(16, 256 and 4,096 for $N$ = 1, 2, 3), not $256^N$ as for byte-level N-grams.",
        fontsize=7.6, family=F, color="#111827")

plt.savefig("figures/Fig02_hex_char_ngram_window.png", bbox_inches="tight", facecolor="white")
