from pathlib import Path
import shutil
import pandas as pd
import pickle as pkl
import functools
import inspect

RUN_NAME = 'run' # The preface for the folder names for storing runs.
TABLE_NAME = 'run_table.csv'

def run_exists(param_dict, output_dir, ignore_missing=False):
    """
    Check to see if a run matching param_dict exists.

    Parameters
    ----------
    param_dict : dict
        A dictionary that contains the key value pairs corresponding to a
        row in a table. This is usually the parameter values that specify
        a run.
    output_dir : str
        The output directory for the runs. The run table will be stored
        in this directory, as well as subdirectories corresponding to
        individual runs.
    ignore_missing : bool
        Whether or not to ignore missing keys. For instance, if a key is
        in the run_table but not in param_dict, but otherwise param_dict
        matches a row of run_table, then this function returns True if
        ignore_missing is True and False if ignore_missing is False.

    Returns
    -------
    bool
    """
    output_dir = Path(output_dir)
    table_path = output_dir/TABLE_NAME
    if not table_path.exists():  # If the table hasn't been created yet.
        return False
    
    param_df = pd.read_csv(table_path, index_col=0, dtype=str)
    same_keys = set(param_dict.keys()) ==  set(param_df.columns)
    if not ignore_missing and not same_keys:
        return False
    missing_cols = set(param_df.columns) - set(param_dict.keys())
    param_df = param_df.drop(columns=missing_cols)
    new_row = pd.DataFrame(param_dict, index=[0], dtype=str)
    merged = pd.merge(param_df, new_row)
    if len(merged) == 0:
        return False
    else:
        return True

def get_run_entry(param_dict, output_dir, prompt_for_user_input=True):
    """
    Get a run ID and directory that corresponds to param_dict. If a
    corresponding row of the run table does not exist, create
    a new row and generate a new directory for the run and return the
    corresponding new ID and new directory.

    Parameters
    ----------
    param_dict : dict
        A dictionary that contains the key value pairs corresponding to a
        row in a table. This is usually the parameter values that specify
        a run.
    output_dir : str
        The output directory for the runs. The run table will be stored
        in this directory, as well as subdirectories corresponding to
        individual runs.

    Returns
    -------
    int
        The number uniquely identifying the run. This is also the index for
        the run in the run table.
    """
    output_dir = Path(output_dir)
    table_path = output_dir/TABLE_NAME
    if not table_path.exists():  # If the table hasn't been created yet.
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        param_df = pd.DataFrame(param_dict, index=[0], dtype=str)
        param_df.index.name = 'index'
        param_df.to_csv(table_path)
        return 0
    
    param_df = pd.read_csv(table_path, index_col=0, dtype=str)
    missing_keys = set(param_dict.keys()) - set(param_df.columns)
    if len(missing_keys) > 0:
        print("""The following keys are in the run table but not in param_dict.
Please specify these keys in param_dict:""")
        print(missing_keys)
        raise ValueError("Missing parameter keys.")

    extra_keys = set(param_dict.keys()) - set(param_df.columns)
    extra_keys_not_set = extra_keys.copy()
    if prompt_for_user_input:
        while len(extra_keys_not_set) > 0:
            new_col_key = extra_keys_not_set.pop()
            new_param_value = input(
    f"""New parameter '{new_col_key}' detected. Please enter value for previous
runs.
Enter value: """)
            new_param_value = str(new_param_value)
            param_df[new_col_key] = new_param_value
    else:
        raise ValueError("""Extra keys specified in param_dict that were not
 previously specified in run table. Please remove
 these keys or set prompt_to_set_values to True to
 set the values for previous runs.""")

    param_dict_row = pd.DataFrame(param_dict, index=[0], dtype=str)
    # This merges while preserving the index
    merged = param_df.reset_index().merge(param_dict_row).set_index('index')
    if len(merged) == 0:
        run_id = max(list(param_df.index)) + 1
        param_df = param_df.append(param_dict_row, ignore_index=True)
        param_df.to_csv(table_path)
    else:
        run_id = merged.index[0]
    return run_id

class Memory:
    def __init__(self, out_dir, run_name='run_'):
        self.out_dir = Path(out_dir)
        self.run_name = run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def cache(self, func):
        signature = inspect.signature(func)
        arg_names = list(signature.parameters.keys())
        kwarg_names_unset = set(arg_names)
        default_kwarg_vals = {
            key: val.default
            for key, val in signature.parameters.items()
            if val.default is not inspect.Parameter.empty
        }
        @functools.wraps(func)
        def memoized_func(*args, **kwargs):
            kwarg_names_unset_local = kwarg_names_unset.copy()
            arg_dict = {}
            for k, arg in enumerate(args):
                arg_dict[arg_names[k]] = arg
                kwarg_names_unset_local.remove(arg_names[k])
            for kwarg in kwargs:
                arg_dict[kwarg] = kwargs[kwarg]
                kwarg_names_unset_local.remove(kwarg)
            for kwarg in kwarg_names_unset_local: 
                arg_dict[kwarg] = default_kwarg_vals[kwarg]
            load = run_exists(arg_dict, self.out_dir)
            run_id = get_run_entry(arg_dict, self.out_dir)
            fdir = self.out_dir/(self.run_name + str(run_id))
            fdir.mkdir(parents=True, exist_ok=True)
            filename = 'function_cache.pkl'
            filepath = fdir/filename
            if load:
                try:
                    with open(filepath, 'rb') as fid:
                        return pkl.load(fid)
                except FileNotFoundError:
                    pass
            # fn_out = {'out_tuple': func(*args, **kwargs)}
            fn_out = func(*args, **kwargs)
            with open(filepath, 'wb') as fid:
                pkl.dump(fn_out, fid, protocol=5)
            return fn_out
        
        return memoized_func

    def clear(self):
        shutil.rmtree(self.out_dir)



if __name__ == '__main__':
    output_dir = Path('output')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    d = {'a': 1, 'b': 2}
    print(run_exists(d, output_dir)) 
    run_id = get_run_entry(d, output_dir)
    print(run_exists(d, output_dir)) 
    run_id = get_run_entry(d, output_dir)
    d = {'a': 2, 'b': 2}
    run_id = get_run_entry(d, output_dir)
    print()
    memory = Memory(output_dir)
    memory.clear()
    
    @memory.cache
    def foo(a, b=3):
        return 2*a + b
    
    foo(1, 2)
    foo(1)
    foo(1, 2)
    print()


    
