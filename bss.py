import matplotlib.pyplot as plt
import generate_data as gd
import ica

# number of components to retain
ica_obj = ica.ICA(gd.X)
inde_X = ica_obj.find_sources()

plt.figure()

models = [gd.X, gd.S, inde_X]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
