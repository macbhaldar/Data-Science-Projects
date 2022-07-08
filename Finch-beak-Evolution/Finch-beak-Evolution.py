# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

# Loading the data
finch_beaks_1975 = pd.read_csv("data/finch_beaks_1975.csv")
finch_beaks_2012 = pd.read_csv("data/finch_beaks_2012.csv")

# cleaning the data for analysis
finch_beaks_1975 = finch_beaks_1975.drop(['band'], axis = 'columns')
finch_beaks_2012 = finch_beaks_2012.drop(['band'], axis = 'columns')

finch_beaks_1975['year'] = "1975"
finch_beaks_2012['year'] = "2012"
finch_beaks_1975.rename(columns={'Beak depth, mm' : 'bdepth','Beak length, mm' : 'blength'}, inplace=True)
finch_beaks_both = pd.concat([finch_beaks_1975,finch_beaks_2012]).reset_index(drop=True)

fortis_f = finch_beaks_both[finch_beaks_both.species == 'fortis'].reset_index(drop=True)
scandens_f = finch_beaks_both[finch_beaks_both.species == 'scandens'].reset_index(drop=True)
finch_beaks_both.info()

markers = {'1975': "s", '2012': "X"}
plt.figure(figsize=(8,6))
sns.scatterplot(x='blength', y='bdepth', style='year', markers=markers, data=fortis_f)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='blength', y='bdepth', style='year', markers=markers, data=scandens_f)
plt.show()

scandens_1975 = finch_beaks_1975[finch_beaks_1975['species']=='scandens']
scandens_2012 = finch_beaks_2012[finch_beaks_2012['species']=='scandens']

# depths of scandens finch beak
scandens_beak_depth_1975 = scandens_1975['bdepth'].reset_index(drop=True)
scandens_beak_depth_2012 = scandens_2012['bdepth'].reset_index(drop=True)

# lengths of scandens finch beak
scandens_beak_length_1975 = scandens_1975['blength'].reset_index(drop=True)
scandens_beak_length_2012 = scandens_2012['blength'].reset_index(drop=True)

# Exploratory Data Analysis

# create bee swarm plot
sns.swarmplot(x='year', y='bdepth', data=scandens_f)

plt.xlabel('year')
plt.ylabel('beak depth (mm)')
plt.show()

# ECDF of the scanden species beak depth
def ecdf(x_data) :
    x = np.sort(x_data)
    y = np.arange(1,len(x)+1) / len(x)
    return x,y

x_1975, y_1975 = ecdf(scandens_beak_depth_1975) #computing ecdf
x_2012, y_2012 = ecdf(scandens_beak_depth_2012)

_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')
_ = plt.margins(0.02)
_ = plt.xlabel('Beak Depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')
plt.show()

# Determining Confidence Interval
def bs_reps(data, func, size=1):
    bs_rep = np.empty(size)
    
    for i in range(size):
        bs_rep[i] = func(np.random.choice(data,size=len(data)))
    return bs_rep
  
# Compute the difference of the mean beak depth reported in 1975 and 2012
mean_diff = np.mean(scandens_beak_depth_2012) - np.mean(scandens_beak_depth_1975)

# Now bootstrap both the depths using mean function for 10000 samples
bs_rep_1975 = bs_reps(scandens_beak_depth_1975, np.mean, size=10000)
bs_rep_2012 = bs_reps(scandens_beak_depth_2012, np.mean, size=10000)

# Compute the difference of the sample means
bootstrap_rep = bs_rep_2012 - bs_rep_1975

# Compute the 95% confidence interval
conf_int = np.percentile(bootstrap_rep, [2.5, 97.5])

print('Difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')

'''
From the ECDF plot and the confidence interval, 
we can confirm that Darwin’s Scandens species beak depth 
increased from 1975 to 2012, confirming the natural selection evolution theory.
'''

# Use bootstrap sampling to compute the difference of means
combined_mean = np.mean(np.concatenate((scandens_beak_depth_1975, scandens_beak_depth_2012)))

# Shifting the two data sets so that they have the same mean
bd_1975_shift = scandens_beak_depth_1975 - np.mean(scandens_beak_depth_1975) + combined_mean
bd_2012_shift = scandens_beak_depth_2012 - np.mean(scandens_beak_depth_2012) + combined_mean

bs_rep_1975_shift = bs_reps(bd_1975_shift, np.mean, size=10000) #bootstrapping
bs_rep_2012_shift = bs_reps(bd_2012_shift, np.mean, size=10000)

bs_shifted_mean_diff = bs_rep_2012_shift - bs_rep_1975_shift

# Calculating p-value
p= np.sum(bs_shifted_mean_diff >= mean_diff) / len(bs_shifted_mean_diff)
print("p-value = ",p)

# Scatter plot of 1975 & 2012 beak length and depth
_ = sns.scatterplot(x= 'blength', y= 'bdepth',  data= scandens_1975)
_ = sns.scatterplot(x= 'blength', y= 'bdepth',  data= scandens_2012)

_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')
plt.show()

# Beak length to depth ratio
ratio_1975 = scandens_beak_length_1975 / scandens_beak_depth_1975
ratio_2012 = scandens_beak_length_2012 / scandens_beak_depth_2012

mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = bs_reps(ratio_2012, np.mean, size=10000)

conf_int_1975 = np.percentile(bs_replicates_1975, [0.5, 99.5])
conf_int_2012 = np.percentile(bs_replicates_2012, [0.5, 99.5])

print('1975: mean ratio =', mean_ratio_1975, 'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012, 'conf int =', conf_int_2012)

y1 = np.full((10000), 1975)
y2 = np.full((10000), 2012)
_ = plt.figure(figsize=(5,4))
_ = plt.plot(mean_ratio_1975, 1975, 'ro', color = 'b')
_ = plt.plot(mean_ratio_2012, 2012, 'ro', color = 'green')
_ = plt.plot(bs_replicates_1975, y1)
_ = plt.plot(bs_replicates_2012, y2)
_ = plt.yticks([1975, 2012])
_ = plt.margins(0.6)
plt.show()
