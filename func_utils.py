import pickle
import os

current_dir = os.getcwd()
json_dir = os.path.join(current_dir, 'json')

def pickle_file(filename, object_name):
    with open(os.path.join(json_dir, filename), 'wb') as f:
        jfile = pickle.dump(object_name, f)
    return jfile

def unpickle_file(filename):
    with open(os.path.join(json_dir, filename), 'rb') as f:
        jfile = pickle.load(f)
    return jfile


def create_project_dirs(project_name, json_dir):
    """
    Creates by default the working directory in the user's "Documents" folder
    and all the nested directories, namely,"Data", "Analysis", "Models",
    where the outputs of the script will be dumped
    
    Parameters
    ----------
    name: str, optional
        The user must give a name to the project folder.

    Returns
    -------
    List of strings
        The list contains the paths of each subdirectory
    """

    sub_dirs = ['Data', 'Models', 'Reports']
    sub_dirs_paths = {} # will be used by each module to store its outputs in the correct directory

    root = os.path.expanduser("~/Documents")
    project_dir = os.path.join(root, project_name)

    # exists_ok prevents the FileExistsError if the directory already exists
    os.makedirs(project_dir, exist_ok=True)

    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(project_dir, sub_dir), exist_ok=True)
        sub_dirs_paths[sub_dir] = os.path.join(project_dir, sub_dir)
            
    pickle_file('sub_dirs_list', sub_dirs_paths)
    pickle_file('project_name', project_name)
    
    print('  Project folder successfuly created!')
    print(f"  Folder name: {project_name}")
    
    return sub_dirs_paths

