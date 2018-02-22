import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

print "##################################################################################"
print "# MSDS603 FUNNEL SIMULATION                                                      #"
print "# Team:  Yimei(Liz) Chen, Fei Liu, Tina(GongTing) Peng, Beiming Liu, Sangyu Shen #"
print "##################################################################################"

##########################################################################################
print "\nN.B. All output image will be saved to the local directory\n"
# Question 1
def UserSim(n, my_lambda):
    sim = list(np.random.exponential(1/float(my_lambda), n)) 
    return sim

print "Question 1"

print "Q1 a:"
np.random.seed(123)

my_sim = UserSim(1000, 2)
my_cum = np.cumsum(my_sim)
x = [0.25,0.5,0.75,1,
     1.25,1.5,1.75,2,2.25,2.5,2.75,3]
y = []
for i in x:
    num = np.sum(np.array(my_sim) >= i)
    y.append(num)

plt.figure(figsize=(16,9))
plt.bar(x, y, width =0.1, color='b')
plt.plot(x, y, color = 'b')
plt.title('Figure 1: Funnel Visualization: lambda = 2')
plt.xlabel('Time t')
plt.ylabel('Number of survived users at time t')
plt.savefig("Q1a.png")

print "Q1a.png saved.\n"


print "Q1 b:"
plt.figure(figsize=(24,13))
lambdas = list(np.round(np.arange(0.2,3.2,0.2),2))
for l in lambdas:
    my_sim = UserSim(1000,l)
    y = []
    for i in x:
        num = np.sum(np.array(my_sim) >= i)
        y.append(num)
    plt.plot(x,y, label = 'lambda: {}'.format(l))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Figure 2: Funnel Visualization: various lambda')
plt.xlabel('Time t')
plt.ylabel('Number of survived users time t')
plt.savefig("Q1b.png")

print "Q1b.png saved.\n"

##########################################################################################
# Question 2
print "Question 2"

print "Q2 b:"
np.random.seed(321)
my_sim = UserSim(n=1000, my_lambda=1)
lambda_hat = 1/np.mean(np.array(my_sim))
print "lambda_hat = %.4f \n" % lambda_hat

print "Q2 c:"
np.random.seed(321)
alpha = 0.05
lambda_hat = []
for i in range(500):
    lambda_hat.append(1/np.mean(np.random.choice(my_sim, len(my_sim))))
    
lower_bound = np.percentile(lambda_hat, alpha * 100)
upper_bound = np.percentile(lambda_hat, (1-alpha) * 100)

print "Using that same sample of 1,000 users and 500 bootstrap,"
print "the 95% confidence interval for the lambda estimate is: "
print "\t Lower Bound: %.4f" %  lower_bound
print "\t Upper Bound: %.4f" %  upper_bound


print "\nQ2 d:"
user_list = [100, 200, 500, 1000, 2000, 5000, 10000]
lambda_hat_list = []
lower_bound_list = []
upper_bound_list = []

for n_user in user_list:
    np.random.seed(n_user) # set seed
    my_sim = UserSim(n=n_user, my_lambda=1) #create simulation list
    lambda_hat_list.append(1/np.mean(np.array(my_sim))) # 
    
    lambda_hat = []
    for i in range(500):
        lambda_hat.append(1/np.mean(np.random.choice(my_sim, len(my_sim))))
    
    lower_bound_list.append(np.percentile(lambda_hat, alpha * 100))
    upper_bound_list.append(np.percentile(lambda_hat, (1-alpha) * 100))

df = pd.DataFrame({'user_list': user_list, 'lambda_estimate': lambda_hat_list, 
              'lower_bound_95%': lower_bound_list, 'upper_bound_95%': upper_bound_list}, 
              columns = ['user_list', 'lambda_estimate', 'lower_bound_95%', 'upper_bound_95%'])

# table
print "Print Table:\n"
print df

# visual
lower_error = np.array(lambda_hat_list) - np.array(lower_bound_list)
upper_error = np.array(upper_bound_list) - np.array(lambda_hat_list)
ci = [lower_error, upper_error]
plt.figure(figsize=(16,9))
line,caps,bars=plt.errorbar(
    user_list,     # X
    lambda_hat_list,    # Y
    yerr=ci,        # Y-errors
    fmt="o--",    # format line like for plot()
    linewidth=3,   # width of plot line
    elinewidth=0.5,# width of error bar line
    ecolor='k',    # color of error bar
    capsize=5,     # cap length for error bar
    capthick=0.5   # cap thickness for error bar
    )

plt.title('Figure 3: Estimated lambda with confidence interval')
plt.xlabel('Number of users')
plt.ylabel('Estimated lambda')

plt.setp(line,label="95% Confidence Interval")#give label to returned line
plt.legend(numpoints=1,             #Set the number of markers in label
           loc=('upper left'))      #Set label location
plt.xlim((50,11000))                 #Set X-axis limits
plt.xticks(user_list, rotation='vertical')               #get only ticks we want
plt.savefig("Q2d.png")
print "\nQ2d.png saved.\n"


##########################################################################################
# Question 3, 4
print "Question 3 and 4"
print "computing......\n"

def HurdleFun(sim, bp):
    user_steps = [] # number of steps that each user complete before exit
    output = []
    bp = np.array(bp)
    
    for user in sim: 
        user_steps.append((user-bp>0).sum())
    
    for i in range(len(bp)+1):
        output.append(sum(np.float64(i) == user_steps))
        
    return output

def F_exp(x, Lambda=1): return 1-np.exp(-Lambda*x)

def EstLam2(Hurdle_output, BP): 
    m = len(Hurdle_output)
    
    return lambda L: \
        Hurdle_output[0]*np.log(F_exp(BP[0], L))+\
        sum([Hurdle_output[i-1]*np.log(F_exp(BP[i-1], L)-F_exp(BP[i-2], L)) for i in range(2,m)])+\
        Hurdle_output[m-1]* (-L* BP[m-2]) # simplify formula to avoid "inf" value

def MaxMLE(Hurdle_output, BP, lambda_list):
    max_mle = -float("Inf")
    lam = 0
    mle = EstLam2(Hurdle_output, BP)
    for l in list(np.arange(.1, 3, .05)):
        likelihood = mle(l)
        if likelihood > max_mle:
            max_mle = likelihood
            lam = l
    return lam

def sim_graph(niter= 1000, n=100, l =1 , bp = [.25, .75]):
    estlam1,estlam2=0,0
    diff = []
    for i in range(niter):
        my_sim = UserSim(n=n, my_lambda=l)
        est1 = 1/np.mean(np.array(my_sim))
        est2 = MaxMLE( HurdleFun(my_sim, bp), bp, list(np.arange(.1, 3, .05)))
        estlam1 += est1
        estlam2 += est2
        diff.append(abs(est1 - est2)) # calculate abs(diff)
    return np.mean(diff)

a = sim_graph(bp = [0.25, 0.75])
b = sim_graph(bp = [0.25, 3])
c = sim_graph(bp = [0.25, 10])

print "The difference between the estimated lambdas using method 1 and 2:"
print "Breakpoints:[0.25, 0.75]: %.4f" % a
print "Breakpoints:[0.25,    3]: %.4f" % b
print "Breakpoints:[0.25,   10]: %.4f" % c

print "\nThe final combined plots to reach conclusion for 4b does not plot by default"
print "since it takes 2 minutes to compute, uncomment the final part of the py file to plot"

####### uncomment to plot #######
# plt.figure(figsize = (16, 12))

# niter= 1000
# n=100
# second = [0.3, 0.5, 0.75, 1,1.5, 2,2.5, 3, 5, 7, 8,9,10, 15]
# lambda_list = [0.2, 0.5, 1, 1.5, 2]
# min_second_point = [] 

# for my_lambda in lambda_list:

#     acc = np.zeros(len(second))
#     for i in range(niter):
#         my_sim = UserSim(n=n, my_lambda= my_lambda)
#         est1 = 1/np.mean(np.array(my_sim))
        
#         diff_lamda =[]
#         for i in second :
#             diff_lamda.append( abs(MaxMLE(HurdleFun(my_sim, [0.25,i]), [0.25,i], list(np.arange(.1, 3, .05))) - est1))
#         acc += np.array(diff_lamda)
        
#     min_second_point.append(second[np.argmin(acc)])
    
    
#     plt.xticks(second, rotation=45)
#     plt.plot(second, acc/niter, label = 'lambda: {}'.format(my_lambda))
#     plt.xlabel('Second breakpoint', font=20)
#     plt.ylabel('Average difference between Lamba1 and Lambda2')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig("Q4b.png")
# print "\nQ4b.png saved.\n"


######### END #########