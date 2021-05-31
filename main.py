'''
Optimization Methods - Mid-Term Project: LASSO regression
Jay Liao (ID: RE6094028)
re6094028@gs.ncku.edu.tw
'''

import os, time, pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from utils import *
from args import init_arguments
from datetime import datetime
from scipy.io import arff
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV

def main(args, model_name):
    print('\n---->', model_name, '\n')
    dataPATH = args.dataPATH + args.dataFN if args.dataPATH[-1] == '/' else args.dataPATH + '/' + args.dataFN
    savePATH = args.savePATH if args.savePATH[-1] == '/' else args.savePATH + '/'
    
    try:
        os.makedirs(savePATH)
    except FileExistsError:
        pass
    
    data = arff.loadarff(dataPATH)
    try:
        df0 = pd.DataFrame(data[0]).drop('sample', axis=1)
    except:
        df0 = pd.DataFrame(data[0])
    df1 = pd.DataFrame(data[1])

    X_tr, X_te, y_tr, y_te = train_test_split(
        df0.drop(args.y_label, axis=1), (df0[args.y_label] == b'A')*1, test_size=args.test_size)
    
    L = args.n_alphas + 1
    D = args.ub_alpha - args.lb_alpha
    if model_name == 'logistic':
        model = LogisticRegression()
        params = {
            'penalty': ['l1'],
            'C': 1 / np.arange(args.lb_alpha, args.ub_alpha, D/L)[1:],
            'random_state': [args.random_state],
            'solver': ['liblinear'],
            'max_iter': [args.max_iter]
        }
    else:
        model = ElasticNet()#Lasso()
        #args.eval_metrics = ['mse', 'accuracy']
        #args.final_metric = 'mse'
        params = {
            'alpha': np.arange(args.lb_alpha, args.ub_alpha, D/L)[1:],
            'random_state': [args.random_state],
            'l1_ratio': [1],
            'max_iter': [args.max_iter]
        }
    model_grid = GridSearchCV(model, params, n_jobs=args.n_jobs, cv=args.cv, refit=False, verbose=args.verbose, scoring=set_eval_metrics(args.eval_metrics))

    # train
    t0 = time.time()
    print(f'Start training lasso {model_name} with {args.cv}-fold cv and grid searching ...')
    model_grid.fit(X_tr, y_tr)
    tDiff = time.time() - t0
    print(f'Finish fitting! Time cost: {tDiff:5.2f} s')
    print()
        
    # save the cv performance
    dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    fn = savePATH + dt + model_name + '_nalphas=' + str(args.n_alphas) + '_cv=' + str(args.cv) + '_cv_performance.txt'
    pd.DataFrame(model_grid.cv_results_).to_csv(fn)
    print('The cv performance is saved as', fn)

    # refit with the tuned hyper-parameters
    if args.final_metric == 'mse':
        best_idx = model_grid.cv_results_['mean_test_' + args.final_metric].argmin()
    else:
        best_idx = model_grid.cv_results_['mean_test_' + args.final_metric].argmax()
    best_params = model_grid.cv_results_['params'][best_idx]
    model = LogisticRegression(**best_params) if model_name == 'logistic' else ElasticNet(**best_params)
    model.fit(X_tr, y_tr)
    print(model)

    # save the testing performance 
    y_hat = model.predict(X_te)
    testing_performance = best_params
    print('\nTesting performance:')
    for m in args.eval_metrics:
        testing_performance[m] = set_eval_metric(m)(y_te, y_hat)
        print(f'{m:10s} {testing_performance[m]:.4f}')
    fn0 = fn.replace('_cv_', '_testing_')
    pd.DataFrame(testing_performance, index=[0]).to_csv(fn0, index=None)
    print('The testing performance is saved as', fn0)
    
    # save coefficients of tuned models
    x = model.coef_[0] if model.coef_.ndim == 2 else model.coef_
    print('# non-zero coefficients:', sum(x > 0))
    fn0 = fn0.replace('performance', 'coef')
    np.savetxt(fn0, x)
    print('\nThe tuned model coefficients are saved as', fn0)

    # save the model
    if args.save_model:
        fn_ = fn.replace('performace', 'model').replace('.txt', '.pkl')
        with open(fn_, 'wb') as f:
            pickle.dump(model_grid, f, pickle.HIGHEST_PROTOCOL)

    # create and save plots
    if args.plot_results:
        x = 1/params['C'] if model_name == 'logistic' else params['alpha']

        ## Non-MSE plot
        plt.figure()
        for m in args.eval_metrics:
            if m.lower() != 'mse':
                y = model_grid.cv_results_['mean_test_' + m]
                yerr = model_grid.cv_results_['std_test_' + m]
                plt.errorbar(x, y, yerr=yerr, label=m)   
        if len(args.eval_metrics) > 1:
            plt.legend()
        #plt.title('Results of ' + str(args.cv) + '-fold Cross-Validation')
        plt.xlabel('Lambda')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid()
        fn1 = fn.replace('performance', 'acc_plot').replace('.txt', '.png')
        plt.savefig(fn1)

        ## MSE plot
        plt.figure()
        m = 'mse'
        y = model_grid.cv_results_['mean_test_' + m]
        yerr = model_grid.cv_results_['std_test_' + m]
        plt.errorbar(x, y, yerr=yerr, label=m)
        #plt.title('MSE Plot of ' + str(args.cv) + '-fold Cross-Validation')
        plt.xlabel('Lambda')
        plt.ylabel('MSE')
        plt.grid()
        fn2 = fn1.replace('acc_plot', 'mse_plot')
        plt.savefig(fn2)
        
        ## Fitting time plot
        y = model_grid.cv_results_['mean_fit_time']
        plt.figure()
        plt.plot(x, y)
        plt.grid()
        plt.xlabel('Lambda')
        plt.ylabel('Fitting Time (seconds)')
        fn3 = fn1.replace('acc_plot', 'fittingtime_plot')
        plt.savefig(fn3)

        ## Coef plot
        x = model.coef_[0] if model.coef_.ndim == 2 else model.coef_
        M = max(x.max(), abs(x.min()))*1.1
        plt.figure()
        #plt.figure(figsize=(10,8))
        plt.stem(x)
        plt.grid()
        plt.ylim(-M, M)
        plt.xlabel('Predictor')
        plt.ylabel('Coefficient')
        fn4 = fn1.replace('acc_plot', 'coef_plot')
        plt.savefig(fn4)

        print('Plots are saved as:')
        print(fn1)
        print(fn2)
        print(fn3)
        print(fn4)
    
if __name__ == '__main__':
    args = init_arguments().parse_args()
    for model_name in args.model_names:
        main(args, model_name)
        