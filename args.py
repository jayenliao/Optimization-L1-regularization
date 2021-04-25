import argparse

def init_arguments():
    parser = argparse.ArgumentParser(prog='Optimization Methods - Mid-Term Project: LASSO regression')
    
    parser.add_argument('--random_state', type=int, default=4028)
    parser.add_argument('--savePATH', type=str, default='./output/')
    parser.add_argument('--dataPATH', type=str, default='./data/')
    parser.add_argument('--dataFN', type=str, default='wave_2_classes_with_irrelevant_attributes.arff')
    parser.add_argument('--y_label', type=str, default='classe', help='Column name of y (dependent variable)')
    parser.add_argument('--test_size', type=float, default=.25)
    parser.add_argument('--model_names', nargs='+', type=str, default=['logistic', 'regression'])
    parser.add_argument('--ub_alpha', type=float, default=1.)
    parser.add_argument('--lb_alpha', type=float, default=0.)
    parser.add_argument('--n_alphas', type=int, default=100, help='How many candidate alphas are going to be tuned?')
    parser.add_argument('--l1_ratio', nargs='+', type=float, default=[1.], help='L1 ratios that are going to be tuned. [0., 1.].')
    parser.add_argument('--max_iter', type=int, default=10000, help='Max. no. of interations')
    parser.add_argument('--cv', type=int, default=10, help='No. of cross-validation')
    parser.add_argument('--eval_metrics', nargs='+', type=str, default=['mse', 'accuracy', 'precision', 'recall', 'f1'])
    parser.add_argument('--final_metric', type=str, default='f1')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--n_jobs', type=int, default=32)
    parser.add_argument('--save_model', action='store_false', default=True)
    parser.add_argument('--plot_results', action='store_false', default=True)
    
    return parser
    