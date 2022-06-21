# save .pickle GAN loss in main_version_03.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle






with open('GAN_loss.pickle', 'rb') as f:
    G_Losses, D_Losses, R_Losses = pickle.load(f)

epochs_count = len(G_Losses)
epochs = list (range(epochs_count))
epochs = [5*(i+1) for i in epochs]

print (epochs)

mpl.rcParams['axes.linewidth'] = 2.5
fig, ax = plt.subplots(figsize = (12, 8))
ax.tick_params(axis='both', which='major', labelsize=20)

plt.xlabel('epochs', fontsize=30)
plt.ylabel('loss', fontsize=30)

plt.plot(epochs, G_Losses, linewidth=10, color = 'red')
plt.plot(epochs, D_Losses, linewidth=10, color = 'blue')
plt.legend(['G Loss', 'D Loss'])

plt.savefig("G_D_Losses.png", dpi=1000)


