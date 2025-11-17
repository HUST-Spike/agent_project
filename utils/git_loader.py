import os
import shutil
from git import Repo, GitCommandError
from typing import Optional

def clone_repo(repo_url: str, local_path: str, branch: Optional[str] = None) -> str:
    """
    克隆一个 Git 仓库到本地。如果文件夹已存在，则先删除再克隆。

    :param repo_url: 仓库的 URL
    :param local_path: 本地存储路径
    :param branch: 要克隆的特定分支 (可选)
    :return: 克隆到本地的路径
    :raises GitCommandError: 如果 Git 克隆失败
    """
    # 如果路径已存在，先清空
    if os.path.exists(local_path):
        print(f"Path '{local_path}' already exists. Removing...")
        try:
            shutil.rmtree(local_path)
        except OSError as e:
            print(f"Error removing directory {local_path}: {e}")
            raise
    
    print(f"Cloning repository from {repo_url} to {local_path}...")
    try:
        if branch:
            Repo.clone_from(repo_url, local_path, branch=branch)
        else:
            Repo.clone_from(repo_url, local_path)
        
        print(f"Successfully cloned repository.")
        return local_path
    except GitCommandError as e:
        print(f"Error cloning repository: {e}")
        raise