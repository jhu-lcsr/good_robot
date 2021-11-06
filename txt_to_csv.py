import numpy as np
import os

# Visualize with PlotJuggler https://github.com/facontidavide/PlotJuggler
log_name = os.path.expanduser('~/src/real_good_robot/logs/2020-01-11-19-54-58/transitions/trial-reward-value.log.txt' )
print('loading: ' + log_name)
kwargs = {'delimiter': ' ', 'ndmin': 2}
log = np.loadtxt(log_name, **kwargs)
# os.path.join(self.transitions_directory, '%s.log.csv' % log_name)
csv_name = log_name.replace('txt', 'csv')
np.savetxt(csv_name, np.squeeze(log), delimiter=', ')
print('Saved: ' + csv_name)
