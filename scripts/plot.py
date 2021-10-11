import joblib
import matplotlib.pyplot as plt
data = joblib.load('data/paths6.pkl')

position = data['observations'][:,12:15]
target = data['trajectory']

position[:, 1] = -position[:, 1]
position[:, 2] = -position[:, 2]

plt.plot(position)
plt.plot(target)

plt.legend(['x','y','z', 'xt', 'yt', 'zt'])
plt.show()