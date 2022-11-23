import sys
import functools
import itertools
import logging
import multiprocessing as mp
import time
import copy
import xlwt
import csv
import pandas as pd
import cvxpy
import numpy as np
import os

# 设置最大线程

from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ASGL:
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=True, num_cores=None, solver='default', max_iters=500):
        """
        Parameters:
            model: model to be fit (accepts 'lm' or 'qr')
            penalization: penalization to use (accepts None, 'lasso', 'gl', 'sgl', 'asgl', 'asgl_lasso', 'asgl_gl',
                          alasso, agl, real_agl_step1, real_agl_step2)
            intercept: boolean, whether to fit the model including intercept or not
            tol:  tolerance for a coefficient in the model to be considered as 0
            lambda1: parameter value that controls the level of shrinkage applied on penalizations
            alpha: parameter value, tradeoff between lasso and group lasso in sgl penalization
            tau: quantile level in quantile regression models
            lasso_weights: lasso weights in adaptive penalizations
            gl_weights: group lasso weights in adaptive penalizations
            parallel: boolean, whether to execute the code in parallel or sequentially
            num_cores: if parallel is set to true, the number of cores to use in the execution. Default is (max - 1)
            solver: solver to be used by CVXPY. default uses optimal alternative depending on the problem
            max_iters: CVXPY parameter. Default is 500

        Returns:
            This is a class definition so there is no return. Main method of this class is fit,  that has no return
            but outputs automatically to _coef.
            ASGL._coef stores a list of regression model coefficients.
        """
        self.valid_models = ['lm', 'qr']
        self.valid_penalizations = ['lasso', 'gl', 'sgl', 'alasso', 'agl',
                                    'asgl', 'asgl_lasso',
                                    'asgl_gl', 'real_agl_step1', 'real_agl_step2']
        self.model = model
        self.penalization = penalization
        self.intercept = intercept
        self.tol = tol
        self.lambda1 = lambda1
        self.alpha = alpha
        self.tau = tau
        self.lasso_weights = lasso_weights
        self.gl_weights = gl_weights
        self.parallel = parallel
        self.num_cores = num_cores
        self.max_iters = max_iters
        self.coef_ = None
        # CVXPY solver parameters
        self.solver_stats = None
        self.solver = solver

    # Model checker related functions #################################################################################

    def _model_checker(self):
        """
        Checks if the input model is one of the valid options:
         - lm for linear models
         - qr for quantile regression models
        """
        if self.model in self.valid_models:
            return True
        else:
            logging.error(f'{self.model} is not a valid model. Valid models are {self.valid_models}')
            return False

    def _penalization_checker(self):
        """
        Checks if the penalization is one of the valid options:
         - lasso for lasso penalization
         - gl for group lasso penalization
         - sgl for sparse group lasso penalization
         - asgl for adaptive sparse group lasso penalization
         - asgl_lasso for an sparse group lasso with adaptive weights in the lasso part
         - asgl_gl for an sparse group lasso with adaptive weights in the group lasso part
         - real_agl for 2020
        """
        if (self.penalization in self.valid_penalizations) or (self.penalization is None):
            return True
        else:
            logging.error(f'{self.penalization} is not a valid penalization. '
                          f'Valid penalizations are {self.valid_penalizations} or None')
            return False

    def _dtype_checker(self):
        """
        Checks if some of the inputs are in the correct format
        """
        response_1 = False
        response_2 = False
        if isinstance(self.intercept, bool):
            response_1 = True
        if isinstance(self.tol, np.float):
            response_2 = True
        response = response_1 and response_2
        return response

    def _input_checker(self):
        """
        Checks that every input parameter for the model solvers has the expected format
        """
        response_list = [self._model_checker(), self._penalization_checker(), self._dtype_checker()]
        return False not in response_list

    # Preprocessing related functions #################################################################################

    def _preprocessing_lambda(self):
        """
        Processes the input lambda1 parameter and transforms it as required by the solver package functions
        """
        n_lambda = None
        lambda_vector = None
        if self.penalization is not None:
            if isinstance(self.lambda1, (np.float, np.int)):
                lambda_vector = [self.lambda1]
            else:
                lambda_vector = self.lambda1
            n_lambda = len(lambda_vector)
        return n_lambda, lambda_vector

    def _preprocessing_alpha(self):
        """
        Processes the input alpha parameter from sgl and asgl penalizations and transforms it as required by the solver
        package functions
        """
        n_alpha = None
        alpha_vector = None
        if 'sgl' in self.penalization:
            if self.alpha is not None:
                if isinstance(self.alpha, (np.float, np.int)):
                    alpha_vector = [self.alpha]
                else:
                    alpha_vector = self.alpha
                n_alpha = len(alpha_vector)
        return n_alpha, alpha_vector

    def _preprocessing_weights(self, weights):
        """
        Converts l_weights into a list of lists. Each list inside l_weights defines a set of weights for a model
        """
        n_weights = None
        weights_list = None
        if self.penalization in ['asgl', 'asgl_lasso',
                                 'asgl_gl', 'alasso', 'agl',
                                 'real_agl_step1', 'real_agl_step2']:
            if weights is not None:
                if isinstance(weights, list):
                    # If weights is a list of lists -> convert to list of arrays
                    if isinstance(weights[0], list):
                        weights_list = [np.asarray(elt) for elt in weights]
                    # If weights is a list of numbers -> store in a list
                    elif isinstance(weights[0], (np.float, np.int)):
                        weights_list = [np.asarray(weights)]
                    else:
                        # If it is a list of arrays, maintain this way
                        weights_list = weights
                # If weights is a ndarray -> store in a list and convert into list
                elif isinstance(weights, np.ndarray):
                    weights_list = [weights]
                if self.intercept:
                    weights_list = [np.insert(elt, 0, 0, axis=0) for elt in weights_list]
                n_weights = len(weights_list)
        return n_weights, weights_list

    def _preprocessing_itertools_param(self, lambda_vector, alpha_vector, lasso_weights_list, gl_weights_list):
        """
        Receives as input the results from preprocessing_lambda, preprocessing_alpha and preprocessing_weights
        Outputs an iterable list of parameter values "param"
        """
        if self.penalization in ['lasso', 'gl', 'real_agl_step1']:
            param = lambda_vector
        elif self.penalization == 'sgl':
            param = itertools.product(lambda_vector, alpha_vector)
        elif self.penalization == 'alasso':
            param = itertools.product(lambda_vector, lasso_weights_list)
        elif self.penalization == 'agl':
            param = itertools.product(lambda_vector, gl_weights_list)
        elif self.penalization == 'real_agl_step2':
            #             print('Show:', lambda_vector, gl_weights_list)
            param = itertools.product(lambda_vector, gl_weights_list)
        elif 'asgl' in self.penalization:
            param = itertools.product(lambda_vector, alpha_vector, lasso_weights_list, gl_weights_list)
        else:
            param = None
            logging.error(f'Error preprocessing input parameters')
        param = list(param)
        return param

    def _preprocessing(self):
        """
        Receives all the parameters of the models and creates tuples of the parameters to be executed in the penalized
        model solvers
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self._input_checker() is False:
            logging.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        # Defines param as None for the unpenalized model
        if self.penalization is None:
            param = None
        else:
            # Reformat parameter vectors
            n_lambda, lambda_vector = self._preprocessing_lambda()
            n_alpha, alpha_vector = self._preprocessing_alpha()
            n_lasso_weights, lasso_weights_list = self._preprocessing_weights(self.lasso_weights)
            n_gl_weights, gl_weights_list = self._preprocessing_weights(self.gl_weights)
            param = self._preprocessing_itertools_param(lambda_vector, alpha_vector, lasso_weights_list,
                                                        gl_weights_list)
        return param

    # CVXPY SOLVER RELATED OPTIONS ###################################################################################

    def _cvxpy_solver_options(self, solver):
        if solver == 'ECOS':
            solver_dict = dict(solver=solver,
                               max_iters=self.max_iters)
        elif solver == 'OSQP':
            solver_dict = dict(solver=solver,
                               max_iter=self.max_iters)
        else:
            solver_dict = dict(solver=solver)
        return solver_dict

    # SOLVERS #########################################################################################################

    def _quantile_function(self, x):
        """
        Quantile function required for quantile regression models.
        """
        return 0.5 * cvxpy.abs(x) + (self.tau - 0.5) * x

    def _num_beta_var_from_group_index(self, group_index):
        """
        Internal function used in group based penalizations 
        (gl, sgl, asgl, asgl_lasso, asgl_gl, real_agl_step1, real_agl_step2)
        """
        group_sizes = []
        beta_var = []
        unique_group_index = np.unique(group_index)
        # Define the objective function
        for idx in unique_group_index:
            group_sizes.append(len(np.where(group_index == idx)[0]))
            beta_var.append(cvxpy.Variable(len(np.where(group_index == idx)[0])))
        return group_sizes, beta_var

    def unpenalized_solver(self, x, y):
        n, m = x.shape
        # If we want an intercept, it adds a column of ones to the matrix x
        if self.intercept:
            m = m + 1
            x = np.c_[np.ones(n), x]
        # Define the objective function
        beta_var = cvxpy.Variable(m)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - x @ beta_var)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function)
        problem = cvxpy.Problem(objective)
        # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
        # If other name is provided, try the name provided
        # If these options fail, try default ECOS, OSQP, SCS options
        try:
            if self.solver == 'default':
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning('Default solver failed. Using alternative options. Check solver and solver_stats for more '
                            'details')
            solver = ['ECOS', 'OSQP', 'SCS']
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if 'optimal' in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning('Optimization problem status failure')
        beta_sol = beta_var.value
        beta_sol[np.abs(beta_sol) < self.tol] = 0
        return [beta_sol]

    def lasso(self, x, y, param):
        """
        Lasso penalized solver
        """
        n, m = x.shape
        # If we want an intercept, it adds a column of ones to the matrix x.
        # Init_pen controls when the penalization starts, this way the intercept is not penalized
        if self.intercept:
            m = m + 1
            x = np.c_[np.ones(n), x]
            init_pen = 1
        else:
            init_pen = 0
        # Define the objective function
        lambda_param = cvxpy.Parameter(nonneg=True)
        beta_var = cvxpy.Variable(m)
        lasso_penalization = lambda_param * cvxpy.norm(beta_var[init_pen:], 1)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - x @ beta_var)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam in param:
            lambda_param.value = lam
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = beta_var.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        logging.debug('Function finished without errors')
        return beta_sol_list

    def gl(self, x, y, group_index, param):
        """
        Group lasso penalized solver
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        lambda_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(objective_function + (lambda_param * group_lasso_penalization))
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam in param:
            lambda_param.value = lam
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def sgl(self, x, y, group_index, param):
        """
        Sparse group lasso penalized solver
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        lasso_penalization = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * cvxpy.norm(beta_var[i], 2)
            lasso_penalization += cvxpy.norm(beta_var[i], 1)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        lasso_param = cvxpy.Parameter(nonneg=True)
        grp_lasso_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(objective_function +
                                   (grp_lasso_param * group_lasso_penalization) +
                                   (lasso_param * lasso_penalization))
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, al in param:
            lasso_param.value = lam * al
            grp_lasso_param.value = lam * (1 - al)
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def alasso(self, x, y, param):
        """
        Lasso penalized solver
        """
        n, m = x.shape
        # If we want an intercept, it adds a column of ones to the matrix x.
        # Init_pen controls when the penalization starts, this way the intercept is not penalized
        if self.intercept:
            m = m + 1
            x = np.c_[np.ones(n), x]
            init_pen = 1
        else:
            init_pen = 0
        # Define the objective function
        l_weights_param = cvxpy.Parameter(m, nonneg=True)
        beta_var = cvxpy.Variable(m)
        lasso_penalization = cvxpy.norm(l_weights_param[init_pen:].T @ cvxpy.abs(beta_var[init_pen:]), 1)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - x @ beta_var)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - x @ beta_var)))
        objective = cvxpy.Minimize(objective_function + lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, lw in param:
            l_weights_param.value = lam * lw
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = beta_var.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        logging.debug('Function finished without errors')
        return beta_sol_list

    def agl(self, x, y, group_index, param):
        """
        Group lasso penalized solver
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        gl_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * gl_weights_param[i] * cvxpy.norm(beta_var[i], 2)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        objective = cvxpy.Minimize(objective_function + group_lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, gl in param:
            gl_weights_param.value = lam * gl
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')

            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def real_agl_step1(self, x, y, group_index, param):
        """
        Adaptive Group lasso penalized solver
        
        Step 1: the weight vector of penalization is np.ones(num_groups,)
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += 1.0 / n * cvxpy.norm(beta_var[i], 2)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        lambda_param = cvxpy.Parameter(nonneg=True)
        objective = cvxpy.Minimize(objective_function + (lambda_param * group_lasso_penalization))
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam in param:
            lambda_param.value = lam
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def real_agl_step2(self, x, y, group_index, param):
        """
        Adaptive Group lasso penalized solver
        
        Step 2: Using the adaptative weight vector from step one.
        """
        n = x.shape[0]
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            # Adds an element (referring to the intercept) to group_index, group_sizes, num groups
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            # Compute model prediction for the intercept with no penalization
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        gl_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            group_lasso_penalization += 1.0 / n * cvxpy.sqrt(gl_weights_param[i]) * cvxpy.norm(beta_var[i], 2)
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        objective = cvxpy.Minimize(objective_function + group_lasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []

        # Solve the problem iteratively for each parameter value
        for lam, gl in param:
            gl_weights_param.value = lam * gl
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    def asgl(self, x, y, group_index, param):
        """
        adaptive sparse group lasso penalized solver
        """
        n, m = x.shape
        # Check th group_index, find the unique groups, count how many vars are in each group (this is the group size)
        group_index = np.asarray(group_index).astype(int)
        unique_group_index = np.unique(group_index)
        group_sizes, beta_var = self._num_beta_var_from_group_index(group_index)
        num_groups = len(group_sizes)
        model_prediction = 0
        alasso_penalization = 0
        a_group_lasso_penalization = 0
        # If the model has an intercept, we calculate the value of the model for the intercept group_index
        # We start the penalization in inf_lim so if the model has an intercept, penalization starts after the intercept
        inf_lim = 0
        if self.intercept:
            group_index = np.append(0, group_index)
            unique_group_index = np.unique(group_index)
            x = np.c_[np.ones(n), x]
            m = m + 1
            group_sizes = [1] + group_sizes
            beta_var = [cvxpy.Variable(1)] + beta_var
            num_groups = num_groups + 1
            model_prediction = x[:, np.where(group_index == unique_group_index[0])[0]] @ beta_var[0]
            inf_lim = 1
        l_weights_param = cvxpy.Parameter(m, nonneg=True)
        gl_weights_param = cvxpy.Parameter(num_groups, nonneg=True)
        for i in range(inf_lim, num_groups):
            model_prediction += x[:, np.where(group_index == unique_group_index[i])[0]] @ beta_var[i]
            a_group_lasso_penalization += cvxpy.sqrt(group_sizes[i]) * gl_weights_param[i] * cvxpy.norm(beta_var[i], 2)
            alasso_penalization += l_weights_param[np.where(group_index ==
                                                            unique_group_index[i])[0]].T @ cvxpy.abs(beta_var[i])
        if self.model == 'lm':
            objective_function = (1.0 / n) * cvxpy.sum_squares(y - model_prediction)
        else:
            objective_function = (1.0 / n) * cvxpy.sum(self._quantile_function(x=(y - model_prediction)))
        objective = cvxpy.Minimize(objective_function +
                                   a_group_lasso_penalization +
                                   alasso_penalization)
        problem = cvxpy.Problem(objective)
        beta_sol_list = []
        # Solve the problem iteratively for each parameter value
        for lam, al, lw, glw in param:
            l_weights_param.value = lw * lam * al
            gl_weights_param.value = glw * lam * (1 - al)
            # Solve the problem. If solver is left as default, try optimal solver sugested by cvxpy.
            # If other name is provided, try the name provided
            # If these options fail, try default ECOS, OSQP, SCS options
            try:
                if self.solver == 'default':
                    problem.solve(warm_start=True)
                else:
                    solver_dict = self._cvxpy_solver_options(solver=self.solver)
                    problem.solve(**solver_dict)
            except (ValueError, cvxpy.error.SolverError):
                logging.warning(
                    'Default solver failed. Using alternative options. Check solver and solver_stats for more '
                    'details')
                solver = ['ECOS', 'OSQP', 'SCS']
                for elt in solver:
                    solver_dict = self._cvxpy_solver_options(solver=elt)
                    try:
                        problem.solve(**solver_dict)
                        if 'optimal' in problem.status:
                            break
                    except (ValueError, cvxpy.error.SolverError):
                        continue
            self.solver_stats = problem.solver_stats
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            beta_sol = np.concatenate([b.value for b in beta_var], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
            beta_sol_list.append(beta_sol)
        return beta_sol_list

    # Calculate coef

    def cal_agl_weight(self, group_index, coef, tol=1e-5):
        """

        Parameters
        ----------
        group_index : array or list
            the group index of each variable .
        coef : array or list
            \tilde{beta} estimated from 1st step agl.
        tol : float
            If the norm of beta < tol, then it equal to 999999.
        
        Returns
        -------
        step2_weight : list.
            the weight vector of penalization in 2nd step function

        """
        group_sizes = []
        step2_weight = []
        unique_group_index = np.unique(group_index)
        for idx in unique_group_index:
            group_sizes.append(len(np.where(group_index == idx)[0]))
        intercept_flag = 1
        # if self.intercept:
        #     intercept_flag = 1
        for i in range(len(group_sizes)):
            begin = np.argwhere(group_index == i + 1)[0][0]
            s2_weight_temp = np.linalg.norm(
                coef[int(intercept_flag + begin):int(intercept_flag + begin + group_sizes[i])]
                , ord=2) ** 2
            if s2_weight_temp >= tol:
                s2_weight_temp = s2_weight_temp ** (-0.5)
            else:
                s2_weight_temp = 9.9999999999e10
            step2_weight.append(s2_weight_temp)
        return step2_weight

    # PARALLEL CODE ###################################################################################################

    # FIT METHOD ######################################################################################################

    def _get_solver_names(self):
        if 'asgl' in self.penalization:
            return 'asgl'
        else:
            return self.penalization

    def fit(self, x, y, group_index=None):
        """
        Main function of the module. Given a model, penalization and parameter values specified in the class definition,
        this function solves the model and produces the regression coefficients
        """
        param = self._preprocessing()
        if self.penalization is None:
            self.coef_ = self.unpenalized_solver(x=x, y=y)
        else:

            if self.penalization in ['lasso', 'alasso']:
                self.coef_ = getattr(self, self._get_solver_names())(x=x, y=y, param=param)
            else:
                print('self._get_solver_names()', self._get_solver_names())
                print('x', x.shape)
                print('y', y.shape)
                print('-------------------------------')
                self.coef_ = getattr(self, self._get_solver_names())(x=x, y=y, param=param,
                                                                     group_index=group_index)
                print('  self.coef_', len(self.coef_))


    # PREDICTION METHOD ###############################################################################################

    def predict(self, x_new):
        """
        To be executed after fitting a model. Given a new dataset, this function produces predictions for that data
        considering the different model coefficients output provided by function fit
        """
        if self.intercept:
            x_new = np.c_[np.ones(x_new.shape[0]), x_new]
        if x_new.shape[1] != len(self.coef_[0]):
            logging.error('Model dimension and new data dimension does not match')
            raise ValueError('Model dimension and new data dimension does not match')
        # Store predictions in a list
        prediction_list = []
        for elt in self.coef_:
            prediction_list.append(np.dot(x_new, elt))
        return prediction_list

    # NUMBER OF PARAMETERS ############################################################################################

    def _num_parameters(self):
        """
        retrieves the number of parameters to be considered in a model
        Output: tuple [num_models, n_lambda, n_alpha, n_l_weights, n_gl_weights] where
        - num_models: total number of models to be solved for the grid of parameters given
        - n_lambda: number of different lambda1 values
        - n_alpha: number of different alpha values
        - n_l_weights: number of different weights for the lasso part of the asgl (or asgl_lasso) penalizations
        - n_gl_weights: number of different weights for the lasso part of the asgl (or asgl_gl) penalizations
        """
        # Run the input_checker to verify that the inputs have the correct format
        if self._input_checker() is False:
            logging.error('incorrect input parameters')
            raise ValueError('incorrect input parameters')
        if self.penalization is None:
            # See meaning of each element in the "else" result statement.
            result = dict(num_models=1,
                          n_lambda=None,
                          n_alpha=None,
                          n_lasso_weights=None,
                          n_gl_weights=None)
        else:
            n_lambda, drop = self._preprocessing_lambda()
            n_alpha, drop = self._preprocessing_alpha()
            n_lasso_weights, drop = self._preprocessing_weights(self.lasso_weights)
            n_gl_weights, drop = self._preprocessing_weights(self.gl_weights)
            list_param = [n_lambda, n_alpha, n_lasso_weights, n_gl_weights]
            list_param_no_none = [elt for elt in list_param if elt is not None]
            num_models = np.prod(list_param_no_none)
            result = dict(num_models=num_models,
                          n_lambda=n_lambda,
                          n_alpha=n_alpha,
                          n_lasso_weights=n_lasso_weights,
                          n_gl_weights=n_gl_weights)
        return result

    def _retrieve_parameters_idx(self, param_index):
        """
        Given an index for the param iterable output from _preprocessing function, this function returns a tupple
        with the index of the value of each parameter.
        Example: Solving an adaptive sparse group lasso model with 5 values for lambda1, 4 values for alpha,
                 3 possible lasso weights and 3 possible group lasso weights yields in a grid search on
                 5*4*3*3=180 parameters.
                 Inputing param_index=120 (out of the 180 possible values)in this function will output the
                 lambda, alpha, and weights index for such value
        If the penalization under consideration does not include any of the required parameters (for example, if we are
        solving an sparse group lasso model, we do not consider adaptive weights), the output regarding the non used
        parameters are set to be None.
        """
        number_parameters = self._num_parameters()
        n_models, n_lambda, n_alpha, n_l_weights, n_gl_weights = [number_parameters[elt] for elt in number_parameters]
        if param_index > n_models:
            string = f'param_index should be smaller or equal than the number of models solved. n_models={n_models}, ' \
                     f'param_index={param_index}'
            logging.error(string)
            raise ValueError(string)
        # If penalization is None, all parameters are set to None
        if self.penalization is None:
            result = [None, None, None, None]
        # If penalization is lasso or gl, there is only one parameter, so param_index = position of that parameter
        elif self.penalization in ['lasso', 'gl', 'real_agl_step1']:
            result = [param_index, None, None, None]
        # If penalization is sgl, there are two parameters and two None
        elif self.penalization == 'sgl':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], parameter_idx[1][0], None, None]
        elif self.penalization == 'alasso':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_l_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, parameter_idx[1][0], None]
        elif self.penalization == 'agl':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_gl_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, None, parameter_idx[1][0]]
        elif self.penalization == 'real_agl_step2':
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_gl_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], None, None, parameter_idx[1][0]]
        else:
            parameter_matrix = np.arange(n_models).reshape((n_lambda, n_alpha, n_l_weights, n_gl_weights))
            parameter_idx = np.where(parameter_matrix == param_index)
            result = [parameter_idx[0][0], parameter_idx[1][0],
                      parameter_idx[2][0], parameter_idx[3][0]]
        return result

    def retrieve_parameters_value(self, param_index):
        """
        Converts the index output from retrieve_parameters_idx into the actual numerical value of the parameters.
        Outputs None if the parameter was not used in the model formulation.
        To be executed after fit method.
        """
        param_index = self._retrieve_parameters_idx(param_index)
        result = [param[idx] if idx is not None else None for idx, param in
                  zip(param_index, [self.lambda1, self.alpha, self.lasso_weights, self.gl_weights])]
        result = dict(lambda1=result[0],
                      alpha=result[1],
                      lasso_weights=result[2],
                      gl_weights=result[3])
        return result


# ERROR CALCULATOR METHOD #############################################################################################

def _quantile_function(y_true, y_pred, tau):
    """
    Quantile function required for error computation
    """
    return (1.0 / len(y_true)) * np.sum(0.5 * np.abs(y_true - y_pred) + (tau - 0.5) * (y_true - y_pred))


def error_calculator(y_true, prediction_list, error_type="MSE", tau=None):
    """
    Computes the error between the predicted value and the true value of the response variable
    """
    error_dict = dict(
        MSE=mean_squared_error,
        MAE=mean_absolute_error,
        MDAE=median_absolute_error,
        QRE=_quantile_function)
    valid_error_types = error_dict.keys()
    # Check that the error_type is a valid error type considered
    if error_type not in valid_error_types:
        raise ValueError(f'invalid error type. Valid error types are {error_dict.keys()}')
    if y_true.shape[0] != len(prediction_list[0]):
        logging.error('Dimension of test data does not match dimension of prediction')
        raise ValueError('Dimension of test data does not match dimension of prediction')
    # For each prediction, store the error associated to that prediction in a list
    error_list = []
    if error_type == 'QRE':
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred, tau=tau))
    else:
        for y_pred in prediction_list:
            error_list.append(error_dict[error_type](y_true=y_true, y_pred=y_pred))
    return error_list


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CvGeneralClass(ASGL):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None):
        # ASGL
        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters)
        # Adaptive weights
        self.weight_technique = weight_technique
        self.weight_tol = weight_tol
        self.lasso_power_weight = lasso_power_weight
        self.gl_power_weight = gl_power_weight
        self.variability_pct = variability_pct
        self.lambda1_weights = lambda1_weights
        self.spca_alpha = spca_alpha
        self.spca_ridge_alpha = spca_ridge_alpha
        # Relative to CV
        self.error_type = error_type
        self.random_state = random_state

    # FIT WEIGHT AND MODEL ############################################################################################

    def fit_weights_and_model(self, x, y, group_index=None):
        if (self.penalization is not None) and \
                (self.penalization in ['alasso', 'agl', 'asgl', 'asgl_lasso', 'asgl_gl']):
            # Compute weights
            weights_class = WEIGHTS(model=self.model, penalization=self.penalization,
                                    tau=self.tau, weight_technique=self.weight_technique,
                                    lasso_power_weight=self.lasso_power_weight,
                                    gl_power_weight=self.gl_power_weight,
                                    variability_pct=self.variability_pct,
                                    lambda1_weights=self.lambda1_weights,
                                    spca_alpha=self.spca_alpha,
                                    spca_ridge_alpha=self.spca_ridge_alpha)
            self.lasso_weights, self.gl_weights = weights_class.fit(x=x, y=y, group_index=group_index)
        # Solve the regression model and obtain coefficients
        self.fit(x=x, y=y, group_index=group_index)


# CROSS VALIDATION CLASS ##############################################################################################


class CV(CvGeneralClass):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None, nfolds=5):
        """
        Parameters:
            All the parameters from ASGL class
            All the parameters from WEIGHTS class
            error_type: error measurement to use. Accepts:
                'MSE': mean squared error
                'MAE': mean absolute error
                'MDAE': mean absolute deviation error
                'QRE': quantile regression error
            random_state: random state value in case reproducible data splits are required
            nfolds: number of folds in which the dataset should be split. Default value is 5
        """
        # CvGeneralClass
        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters, weight_technique, weight_tol, lasso_power_weight,
                         gl_power_weight, variability_pct, lambda1_weights, spca_alpha, spca_ridge_alpha, error_type,
                         random_state)
        # Relative to cross validation / train validate / test
        self.nfolds = nfolds

    # SPLIT DATA METHODS ##############################################################################################

    def _cross_validation_split(self, nrows, split_index=None):
        """
        Split data based on kfold or group kfold cross validation
        :param nrows: number of rows in the dataset
        :param split_index: Group structure of observations used in GroupKfold. 
                            same length as nrows
        """
        if split_index is None:
            # Randomly generate index
            data_index = np.random.choice(nrows, nrows, replace=False)
            # Split data into k folds
            k_folds = KFold(n_splits=self.nfolds).split(data_index)
        else:
            data_index = np.arange(0, nrows)
            k_folds = GroupKFold(n_splits=self.nfolds).split(X=data_index, groups=split_index)
        # List containing zips of (train, test) indices
        response = [(data_index[train], data_index[validate]) for train, validate in list(k_folds)]
        return response

    # CROSS VALIDATION ################################################################################################

    def _one_step_cross_validation(self, x, y, group_index=None, zip_index=None):
        # Given a zip (one element from the cross_validation_split function) retrieve train / test splits
        train_index, test_index = zip_index
        x_train, x_test = (x[train_index, :], x[test_index, :])
        y_train, y_test = (y[train_index], y[test_index])
        self.fit_weights_and_model(x=x_train, y=y_train, group_index=group_index)
        predictions = self.predict(x_new=x_test)
        error = error_calculator(y_true=y_test, prediction_list=predictions, error_type=self.error_type,
                                 tau=self.tau)
        return error

    def cross_validation(self, x, y, group_index=None, split_index=None):
        error_list = []
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        cv_index = self._cross_validation_split(nrows=x.shape[0], split_index=split_index)
        for zip_index in cv_index:
            error = self._one_step_cross_validation(x, y, group_index=group_index, zip_index=zip_index)
            error_list.append(error)
        return np.array(error_list).T  # Each row is a model, each column is a k fold split


class TVT(CvGeneralClass):
    def __init__(self, model, penalization, intercept=True, tol=1e-5, lambda1=1, alpha=0.5, tau=0.5,
                 lasso_weights=None, gl_weights=None, parallel=False, num_cores=None, solver=None, max_iters=500,
                 weight_technique='pca_pct', weight_tol=1e-4, lasso_power_weight=1, gl_power_weight=1,
                 variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5, spca_ridge_alpha=1e-2, error_type='MSE',
                 random_state=None, train_pct=0.05, validate_pct=0.05, train_size=None, validate_size=None):

        super().__init__(model, penalization, intercept, tol, lambda1, alpha, tau, lasso_weights, gl_weights, parallel,
                         num_cores, solver, max_iters, weight_technique, weight_tol, lasso_power_weight,
                         gl_power_weight, variability_pct, lambda1_weights, spca_alpha, spca_ridge_alpha, error_type,
                         random_state)
        # Relative to / train validate / test
        self.train_pct = train_pct
        self.validate_pct = validate_pct
        self.train_size = train_size
        self.validate_size = validate_size

    # TRAIN VALIDATE TEST SPLIT #######################################################################################

    def _train_validate_test_split(self, nrows):
        """
        Splits data into train validate and test. Takes into account the random state for consistent repetitions
        :param nrows: Number of observations
        :return:  A three items object including train index, validate index and test index
        """
        # Randomly generate index
        data_index = np.random.choice(nrows, nrows, replace=False)
        if self.train_size is None:
            self.train_size = int(round(nrows * self.train_pct))
        if self.validate_size is None:
            self.validate_size = int(round(nrows * self.validate_pct))
        # List of 3 elements of size train_size, validate_size, remaining_size
        split_index = np.split(data_index, [self.train_size, self.train_size + self.validate_size])
        return split_index

    # TRAIN VALIDATE TEST #############################################################################################

    def _train_validate(self, x, y, group_index=None):
        """
        Split the data into train validate and test. Fit an adaptive lasso based model on the train split. Obtain
        predictions and compute error using validation set.
        :param x: data matrix x
        :param y: response vector y
        :param group_index: group index of predictors in data matrix x
        :return: Validate error and test index
        """
        # Define random state if required
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Split data
        split_index = self._train_validate_test_split(nrows=x.shape[0])
        x_train, x_validate, drop = [x[idx, :] for idx in split_index]
        y_train, y_validate, drop = [y[idx] for idx in split_index]
        test_index = split_index[2]
        # Solve models
        self.fit_weights_and_model(x=x_train, y=y_train, group_index=group_index)
        # Obtain predictions and compute validation error
        predictions = self.predict(x_new=x_validate)
        validate_error = error_calculator(y_true=y_validate, prediction_list=predictions,
                                          error_type=self.error_type, tau=self.tau)
        return validate_error, test_index

    def _tv_test(self, x, y, validate_error, test_index):
        """
        Given a validate error and a test index obtains the final test_error and stores the optimal parameter values and
        optimal betas
        :param x: data matrix x
        :param y: response vector y
        :param validate_error: Validation error computed in __train_validate()
        :param test_index: Test index
        :return: optimal_betas, optimal_parameters, test_error
        """
        # Select the minimum error
        minimum_error_idx = np.argmin(validate_error)
        # Select the parameters index associated to mininum error values
        optimal_parameters = self.retrieve_parameters_value(minimum_error_idx)
        # Minimum error model
        optimal_betas = self.coef_[minimum_error_idx]
        test_prediction = [self.predict(x_new=x[test_index, :])[minimum_error_idx]]
        test_error = error_calculator(y_true=y[test_index], prediction_list=test_prediction,
                                      error_type=self.error_type, tau=self.tau)[0]
        return optimal_betas, optimal_parameters, test_error

    def train_validate_test(self, x, y, group_index=None):
        """
        Runs functions __train_validate() and __tv_test(). Stores results in a dictionary
        :param x: data matrix x
        :param y: response vector y
        :param group_index: group index of predictors in data matrix x
        """
        validate_error, test_index = self._train_validate(x, y, group_index)
        optimal_betas, optimal_parameters, test_error = self._tv_test(x, y, validate_error, test_index)
        result = dict(
            optimal_betas=optimal_betas,
            optimal_parameters=optimal_parameters,
            test_error=test_error)
        return result


# TRAIN TEST SPLIT ###################################################################################################

def train_test_split(nrows, train_size=None, train_pct=0.7, random_state=None):
    """
    Splits data into train / test. Takes into account random_state for future consistency.
    """
    # Define random state if required
    if random_state is not None:
        np.random.seed(random_state)
    data_index = np.random.choice(nrows, nrows, replace=False)
    if train_size is None:
        train_size = int(round(nrows * train_pct))
    # Check that nrows is larger than train_size
    if nrows < train_size:
        logging.error(f'Train size is too large. Input number of rows:{nrows}, current train_size: {train_size}')
    # List of 2 elements of size train_size, remaining_size (test)
    split_index = np.split(data_index, [train_size])
    train_idx, test_idx = [elt for elt in split_index]
    return train_idx, test_idx


class WEIGHTS:
    def __init__(self, model='lm', penalization='asgl', tau=0.5, weight_technique='pca_pct', weight_tol=1e-4,
                 lasso_power_weight=1, gl_power_weight=1, variability_pct=0.9, lambda1_weights=1e-1, spca_alpha=1e-5,
                 spca_ridge_alpha=1e-2):
        """
        Parameters:
            model: model to be fit using these weights (accepts 'lm' or 'qr')
            penalization: penalization to use ('asgl', 'asgl_lasso', 'asgl_gl')
            tau: quantile level in quantile regression models
            weight_technique: weight technique to use for fitting the adaptive weights. Accepts 'pca_1', 'pca_pct',
                    'pls_1', 'pls_pct', 'unpenalized_lm', 'unpenalized_qr', 'spca'
            weight_tol: Tolerance value used for avoiding ZeroDivision errors
            lasso_power_weight: parameter value, power at which the lasso weights are risen
            gl_power_weight: parameter value, power at which the group lasso weights are risen
            variability_pct: parameter value, percentage of variability explained by pca or pls components used in
                    'pca_pct', 'pls_pct' and 'spca'
            lambda1_weights: in case lasso is used as weight calculation alternative, the value for lambda1
            spca_alpha: sparse PCA parameter
            spca_ridge_alpha: sparse PCA parameter
        Returns:
            This is a class definition so there is no return. Main method of this class is fit, that returns adaptive
            weights computed based on the class input parameters.
        """
        self.valid_penalizations = ['alasso', 'agl', 'asgl', 'asgl_lasso', 'asgl_gl']
        self.model = model
        self.penalization = penalization
        self.tau = tau
        self.weight_technique = weight_technique
        self.weight_tol = weight_tol
        self.lasso_power_weight = lasso_power_weight
        self.gl_power_weight = gl_power_weight
        self.variability_pct = variability_pct
        self.lambda1_weights = lambda1_weights
        self.spca_alpha = spca_alpha
        self.spca_ridge_alpha = spca_ridge_alpha

    # PREPROCESSING ###################################################################################################

    def _preprocessing(self, power_weight):
        if isinstance(power_weight, (np.int, np.float)):
            power_weight = [power_weight]
        return power_weight

    # WEIGHT TECHNIQUES ###############################################################################################

    def _pca_1(self, x, y):
        """
        Computes the adpative weights based on the first principal component
        """
        pca = PCA(n_components=1)
        pca.fit(x)
        tmp_weight = np.abs(pca.components_).flatten()
        return tmp_weight

    def _pca_pct(self, x, y):
        """
        Computes the adpative weights based on principal component analysis
        """
        # If var_pct is equal to one, the algorithm selects just 1 component, not 100% of the variability.
        if self.variability_pct == 1:
            var_pct2 = np.min((x.shape[0], x.shape[1]))
        else:
            var_pct2 = self.variability_pct
        pca = PCA(n_components=var_pct2)
        # t is the matrix of "scores" (the projection of x into the PC subspace)
        # p is the matrix of "loadings" (the PCs, the eigen vectors)
        t = pca.fit_transform(x)
        p = pca.components_.T
        # Solve an unpenalized qr model using the obtained PCs
        unpenalized_model = ASGL(model=self.model, penalization=None, intercept=True, tau=self.tau)
        unpenalized_model.fit(x=t, y=y)
        beta_qr = unpenalized_model.coef_[0][1:]  # Remove intercept
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p, beta_qr)).flatten()
        return tmp_weight

    def _pls_1(self, x, y):
        """
        Computes the adpative weights based on the first partial least squares component
        """
        # x_loadings_ is the pls equivalent to the PCs
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(x, y)
        tmp_weight = np.abs(pls.x_rotations_).flatten()
        return tmp_weight

    def _pls_pct(self, x, y):
        """
        Computes the adpative weights based on partial least squares
        """
        total_variance_in_x = np.sum(np.var(x, axis=0))
        pls = PLSRegression(n_components=np.min((x.shape[0], x.shape[1])), scale=False)
        pls.fit(x, y)
        variance_in_pls = np.var(pls.x_scores_, axis=0)
        fractions_of_explained_variance = np.cumsum(variance_in_pls / total_variance_in_x)
        # Update variability_pct
        self.variability_pct = np.min((self.variability_pct, np.max(fractions_of_explained_variance)))
        n_comp = np.argmax(fractions_of_explained_variance >= self.variability_pct) + 1
        pls = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(x, y)
        tmp_weight = np.abs(np.asarray(pls.coef_).flatten())
        return tmp_weight

    def _unpenalized(self, x, y):
        """
        Only for low dimensional frameworks. Computes the adpative weights based on unpenalized regression model
        """
        unpenalized_model = ASGL(model=self.model, penalization=None, intercept=True, tau=self.tau)
        unpenalized_model.fit(x=x, y=y)
        tmp_weight = np.abs(unpenalized_model.coef_[0][1:])  # Remove intercept
        return tmp_weight

    def _sparse_pca(self, x, y):
        """
        Computes the adpative weights based on sparse principal component analysis.
        
        """
        # Compute sparse pca
        x_center = x - x.mean(axis=0)
        total_variance_in_x = np.sum(np.var(x, axis=0))
        spca = SparsePCA(n_components=np.min((x.shape[0], x.shape[1])), alpha=self.spca_alpha,
                         ridge_alpha=self.spca_ridge_alpha)
        t = spca.fit_transform(x_center)
        p = spca.components_.T
        # Obtain explained variance using spca as explained in the original paper (based on QR decomposition)
        t_spca_qr_decomp = np.linalg.qr(t)
        # QR decomposition of modified PCs
        r_spca = t_spca_qr_decomp[1]
        t_spca_variance = np.diag(r_spca) ** 2 / x.shape[0]
        # compute variance_ratio
        fractions_of_explained_variance = np.cumsum(t_spca_variance / total_variance_in_x)
        # Update variability_pct
        self.variability_pct = np.min((self.variability_pct, np.max(fractions_of_explained_variance)))
        n_comp = np.argmax(fractions_of_explained_variance >= self.variability_pct) + 1
        unpenalized_model = ASGL(model=self.model, penalization=None, intercept=True, tau=self.tau)
        unpenalized_model.fit(x=t[:, 0:n_comp], y=y)
        beta_qr = unpenalized_model.coef_[0][1:]
        # Recover an estimation of the beta parameters and use it as weight
        tmp_weight = np.abs(np.dot(p[:, 0:n_comp], beta_qr)).flatten()
        return tmp_weight

    def _lasso(self, x, y):
        lasso_model = ASGL(model=self.model, penalization='lasso', lambda1=self.lambda1_weights, intercept=True,
                           tau=self.tau)
        lasso_model.fit(x=x, y=y)
        tmp_weight = np.abs(lasso_model.coef_[0][1:])  # Remove intercept
        return tmp_weight

    def _weight_techniques_names(self):
        return '_' + self.weight_technique

    def _lasso_weights_calculation(self, tmp_weight):
        self.lasso_power_weight = self._preprocessing(self.lasso_power_weight)
        lasso_weights = [1 / (tmp_weight ** elt + self.weight_tol) for elt in self.lasso_power_weight]
        return lasso_weights

    def _gl_weights_calculation(self, tmp_weight, group_index):
        self.gl_power_weight = self._preprocessing(self.gl_power_weight)
        unique_index = np.unique(group_index)
        gl_weights = []
        for glpw in self.gl_power_weight:
            tmp_list = [1 / (np.linalg.norm(tmp_weight[np.where(group_index == unique_index[i])[0]], 2) ** glpw +
                             self.weight_tol) for i in range(len(unique_index))]
            tmp_list = np.asarray(tmp_list)
            gl_weights.append(tmp_list)
        return gl_weights

    def fit(self, x, y=None, group_index=None):
        """
        Main function of the module, given the input specified in the class definition, this function computes
        the specified weights.
        """
        tmp_weight = getattr(self, self._weight_techniques_names())(x=x, y=y)
        if self.penalization == 'alasso':
            lasso_weights = self._lasso_weights_calculation(tmp_weight)
            gl_weights = None
        elif self.penalization == 'agl':
            lasso_weights = None
            gl_weights = self._gl_weights_calculation(tmp_weight, group_index)
        elif self.penalization == 'asgl_lasso':
            lasso_weights = self._lasso_weights_calculation(tmp_weight)
            gl_weights = np.ones(len(np.unique(group_index)))
        elif self.penalization == 'asgl_gl':
            lasso_weights = np.ones(x.shape[1])
            gl_weights = self._gl_weights_calculation(tmp_weight, group_index)
        elif self.penalization == 'asgl':
            lasso_weights = self._lasso_weights_calculation(tmp_weight)
            gl_weights = self._gl_weights_calculation(tmp_weight, group_index)
        else:
            lasso_weights = None
            gl_weights = None
            logging.error(f'Not a valid penalization for weight calculation. Valid penalizations '
                          f'are {self.valid_penalizations}')
        return lasso_weights, gl_weights


def cal_aic(n, mse, beta):
    """
    

    Parameters
    ----------
    n : INT
        # sample size.
    mse : LIST (list of array)
        different lambda for different lambda.
    beta :LIST (list of array)
        different beta for different lambda.

    Returns
    -------
    min_aic_dict : DICT
        {'MinIndex':i, 'MinValue': min_aic}.
    aic_array: array
        different aic for different lambda.

    """

    number_of_lambda = len(beta)
    aic_array = np.array([])
    for i in range(number_of_lambda):
        beta_list = beta[i]
        k_nonzero = np.count_nonzero(beta_list != 0)
        aic = 2 * k_nonzero + n * np.log(mse[i])
        aic_array = np.append(aic_array, aic)
    min_aic_dict = {'MinIndex': np.argmin(aic_array),
                    'MinValue': np.min(aic_array)}
    return min_aic_dict, aic_array


def cal_bic(n, mse, beta):
    number_of_lambda = len(beta)
    print('number_of_lambda', number_of_lambda)
    bic_array = np.array([])
    for i in range(number_of_lambda):
        beta_list = beta[i]
        k_nonzero = np.count_nonzero(beta_list != 0)
        print('k_nonzero', k_nonzero)
        bic = k_nonzero * np.log(n) + n * (np.log(mse[i]))
        bic_array = np.append(bic_array, bic)
    min_bic_dict = {'MinIndex': np.argmin(bic_array),
                    'MinValue': np.min(bic_array)}
    print('np.argmin(bic_array)', np.argmin(bic_array))
    return min_bic_dict, bic_array


def cal_bic2006(x_trans, y, y_pred, group_index, beta, incep_flag):
    n = x_trans.shape[0]
    number_of_lambda = len(beta)
    unique_index = np.unique(group_index)
    num_each_group = np.empty(shape=(len(unique_index),))
    for i in range(len(unique_index)):
        num_each_group[i] = len(np.argwhere(i + 1 == group_index))
    bic2006_array = np.array([])
    betaOLS = np.dot(np.linalg.inv(np.dot(x_trans.T, x_trans)),
                     np.dot(x_trans.T, y))
    for i in range(number_of_lambda):
        df_i = 0
        var = np.var(y_pred[i])
        beta_i = beta[i]
        for j in range(len(unique_index)):
            begin = np.where(group_index == unique_index[j])[0][0]
            end = begin + num_each_group[j]
            beta_j_norm = np.linalg.norm(beta_i[int(incep_flag + begin):int(incep_flag + end)])
            indic_beta_j = (beta_j_norm > 0) + 0
            betaOLS_j_norm = np.abs(betaOLS[j])
            second_term_j = beta_j_norm / betaOLS_j_norm * (num_each_group[j] - 1)
            temp_df = indic_beta_j + second_term_j
            df_i += temp_df
        C_p = np.linalg.norm(y - y_pred[i]) ** 2 / var - n + 2 * df_i
        bic2006_array = np.append(bic2006_array, C_p)

    min_bic2006_dict = {'MinIndex': np.argmin(bic2006_array),
                        'MinValue': np.min(bic2006_array)}

    return min_bic2006_dict, bic2006_array


#### Sort data and get indexes ####

def sortdata(array):
    # print('array',array)
    sort_array = np.sort(array)
    # print('sort_array', sort_array)
    sort_index = []
    for i in array:
        temp = np.argwhere(sort_array == i)[0][0]
        sort_index.append(temp)
    # print('sort_index', sort_index)
    return sort_index


#### Rank Transform ####

def rank_trans(C):
    C_qt = C.copy()
    for i in range(C_qt.shape[1]):
        temp_C = C[:, i]
        temp_C_index = sortdata(temp_C)
        size_temp_C = temp_C.size
        # print('size_temp_C', size_temp_C)
        for j in range(len(temp_C_index)):
            # print(temp_C_index[j])
            C_qt[j, i] = (temp_C_index[j] + 1) / (size_temp_C + 1)
    return C_qt


#### Calculate p_k(c) function

def pfsort(characteristic, L):
    len_interval = 1 / L
    knots = np.arange(0, 1, len_interval)
    # print('knots', knots)
    p_k = np.empty(L + 2, )
    p_k[0] = 1
    p_k[1] = characteristic
    p_k[2] = characteristic ** 2
    for i in range(3, len(p_k)):
        p_k[i] = max(0, (characteristic - knots[i - 2])) ** 2
    # print(p_k)
    return p_k


def gen_P_matrix(tilde_C, L):
    row, col = tilde_C.shape
    # print(' row, col',  row, col, L)
    P_matrix = np.empty(shape=(row, col * (L + 2)))
    # print('P_matrix', P_matrix.shape)
    for i in range(row):
        temp = np.empty(col * (L + 2), )
        for j in range(col):
            tempj = pfsort(tilde_C[i, j], L)
            temp[j * (L + 2): (j + 1) * (L + 2)] = tempj
            # print('j * (L + 2): (j + 1) * (L + 2)', j * (L + 2), (j + 1) * (L + 2))
        P_matrix[i, :] = np.array([temp])
    return P_matrix

    #### Generate group indexes, which are relative to knots selection ####


def gen_group_index(C, L):
    col = C.shape[1]
    # print('gen_group_index', C.shape)
    group_index = np.empty(np.shape(C)[1] * (L + 2), )
    j = 1
    for i in range(col):
        group_index[(i) * (L + 2): (i + 1) * (L + 2)] = j
        j += 1
    return group_index


def process_x_matrix(x, L):
    x_rank= rank_trans(x)
    x_trans_p = gen_P_matrix(x_rank, L)
    return x_trans_p, x_rank,


def dict_to_csv(dict, filename, newline='', wt_type='w'):
    with open(filename, wt_type) as f:
        [f.write('{0},  {1}\n'.format(key, value)) for key, value in dict.items()]


#### Main Test ####
def two_step_agl_main(x, x_out, x_trans, y, y_out, L,
                      lambda1_vec, lambda2_vec, error_type1, error_type2,
                      criterion='bic2006', mycores=None):
    start_time = time.time()
    print('------x_trans', x_trans.shape)
    group_index = gen_group_index(x_trans, L)
    print('group_index', group_index.shape)
    # print(group_index)

    x_row = x.shape[0]

    step1 = ASGL(model='lm', penalization='real_agl_step1', lambda1=lambda1_vec,
                 parallel=True, num_cores=mycores)  # defined the model

    # print('group_index', group_index)
    step1.fit(x, y, group_index=group_index)
    print('step_1 x', x.shape)
    print('step_1 y', y.shape)

    y_pred1 = step1.predict(x)
    print('y_pred1', len(y_pred1))
    # print(y_pred1)
    # print(y)
    step1_model_error = error_calculator(y_true=y, prediction_list=y_pred1, error_type=error_type1, tau=None)
    print('step1_model_error', step1_model_error)

    # Step 1
    model1 = step1.model  # linear model
    penalization1 = step1.penalization  # sparse group lasso penalization
    parallel1 = step1.parallel  # Code executed in parallel
    # error_type1 = 'MSE'  # Error measurement considered.
    all_coef1 = step1.coef_
    print('step1.coef_', len(all_coef1))
    num_models1 = len(lambda1_vec)
    print(f'We define a grid of {num_models1} models.')

    # choose the lambda from the 6
    if criterion == 'bic':
        print('-----I am BIC -------')
        print('x_row', x_row)
        print('step1_model_error', step1_model_error)
        print('all_coef1', len(all_coef1))
        bic_sel_result = cal_bic(n=x_row, mse=step1_model_error, beta=all_coef1)
        print('bic_sel_result', bic_sel_result[0]['MinIndex'])
        optimal_lambda1 = lambda1_vec[bic_sel_result[0]['MinIndex']]
        coef1 = step1.coef_[bic_sel_result[0]['MinIndex']]
        # print('coef1', coef1)

    # Step 2
    print('-----------------lets run step 2-------------------------')
    print('tol', step1.tol)
    print('group_index', group_index)
    step2_gl_weights = step1.cal_agl_weight(group_index=group_index, coef=coef1, tol=step1.tol)
    print('step2_gl_weights', len(step2_gl_weights))

    model2 = 'lm'  # linear model
    penalization2 = 'real_agl_step2'  # two-step group lasso penalization
    parallel2 = True  # Code executed in parallel
    # error_type2 = 'MSE'  # Error measurement considered.

    step2 = ASGL(model=model2, penalization=penalization2,
                 lambda1=lambda2_vec,
                 intercept=True,
                 parallel=True,
                 gl_weights=step2_gl_weights,
                 num_cores=mycores
                 )

    step2.fit(x, y, group_index=group_index)

    y_pred2_in = step2.predict(x)
    # In sample
    step2_model_error_in = error_calculator(y_true=y, prediction_list=y_pred2_in, error_type=error_type1, tau=None)

    num_models2 = len(lambda2_vec)
    print(f'We define a grid of {num_models2} models.')
    all_coef2 = step2.coef_

    if criterion == 'bic':
        bic_sel_result2 = cal_bic(n=x_row, mse=step2_model_error_in, beta=all_coef2)
        optimal_lambda2 = lambda2_vec[bic_sel_result2[0]['MinIndex']]
        coef2 = step2.coef_[bic_sel_result2[0]['MinIndex']]
        min_error_in = step2_model_error_in[bic_sel_result2[0]['MinIndex']]
        minimum_error_idx2 = bic_sel_result2[0]['MinIndex']

    # Out of sample error
    opt_coef = coef2

    if step2.intercept:
        x_out = np.c_[np.ones(x_out.shape[0]), x_out]
    y_pred2_out = np.dot(x_out, opt_coef)

    y_pred2_out_list = [y_pred2_out]
    print('y_pred2_out', y_pred2_out.shape)
    print('y_out', y_out.shape)

    np.savetxt("y_pred2_out.csv", y_pred2_out.shape, delimiter=",")
    np.savetxt("y_out.csv", y_out.shape, delimiter=",")

    min_error_out = error_calculator(y_true=y_out, prediction_list=y_pred2_out_list, error_type=error_type2, tau=None)
    print(f'The second step of AGL select lambda={optimal_lambda2}')
    print(f'Finished with no error. Execution time: {time.time() - start_time}')

    return coef2, min_error_in, min_error_out, y_pred2_in[minimum_error_idx2], y_pred2_out
