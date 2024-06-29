import base64

import loguru
import requests

from util import *

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
}


# 获取某个组织所有的仓库信息
def get_repos(org_name, per_page=200):
    url = f'https://api.github.com/orgs/{org_name}/repos'
    response = requests.get(url, headers=GITHUB_HEADERS, params={'per_page': per_page, 'page': 0})
    if response.status_code == 200:
        repos = response.json()
        repo_count = len(repos)
        if repo_count == per_page:
            raise ValueError(F"指定查找 {per_page} 个地址，但实际上可能超过这个数，建议调大 per_page 参数")

        loguru.logger.info(f'Fetched {len(repos)} repositories for {org_name}.')
        return repos
    else:
        loguru.logger.error(f"Error fetching repositories: {response.status_code}")
        loguru.logger.error(response.text)
        return None


# 拉取某个仓库的指定文件，默认是 readme
def fetch_repo_file(org_name, repo_name, filename="readme", export_dir=None,
                    output_filename="README.md"):
    url = f'https://api.github.com/repos/{org_name}/{repo_name}/{filename}'
    response = requests.get(url, headers=GITHUB_HEADERS)
    if response.status_code == 200:
        readme_content = response.json()['content']
        # 解码base64内容
        readme_content = base64.b64decode(readme_content).decode('utf-8')
        # 使用 export_dir 保存文件
        if export_dir is not None:
            repo_dir = os.path.join(export_dir, repo_name)
            if not os.path.exists(repo_dir):
                os.makedirs(repo_dir)
            # noinspection PyTypeChecker
            readme_path = os.path.join(repo_dir, output_filename)
            with open(readme_path, 'w', encoding='utf-8') as file:
                file.write(readme_content)

        return readme_content
    else:
        loguru.logger.error(f"Error fetching README for {repo_name}: {response.status_code}")
        loguru.logger.error(response.text)
        return None


def fetch_all_org_readme_files():
    # 配置组织名称
    org_name = 'datawhalechina'
    repos = get_repos(org_name)

    # 拉取每个仓库的README
    if repos:
        for repo in tqdm(repos):
            repo_name = repo['name']

            fetch_repo_file(org_name, repo_name,
                            export_dir=r"D:\PycharmProjects\xiebo\diantou\bigdata\datawhalechina\readme_db")
            time.sleep(1)


def main():
    # fetch_all_org_readme_files()

    pass


if __name__ == '__main__':
    main()
