import pandas as pd
import yaml

def load_config(arquivo_config: str = "src/config.yaml") -> dict:
    """Carrega as configurações do projeto a partir de um arquivo YAML.

    Args:
        arquivo_config (str): Caminho para o arquivo de configuração. 
            Padrão é "config.yaml".

    Returns:
        dict: Dicionário contendo as configurações de datasets e parâmetros globais.

    """
    with open(arquivo_config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_datasets(config: dict) -> dict[str, pd.DataFrame]:
    """Lê os arquivos Parquet consolidados e os armazena em um dicionário de DataFrames.

    Esta função percorre os datasets definidos no arquivo de configuração,
    tenta realizar a leitura utilizando o motor (engine) especificado e
    centraliza o tratamento de erros de I/O.

    Args:
        config (dict): Dicionário de configuração carregado via `load_config`.

    Returns:
        Dict[str, pd.DataFrame]: Um dicionário onde as chaves são os nomes curtos 
            do dataset ('sim', 'sinasc') e os valores são os respectivos DataFrames do Pandas.
            Se um arquivo falhar no carregamento, ele não será incluído no dicionário.
    """
    dataframes = {}
    engine = config['settings']['engine']

    for key, info in config['datasets'].items():
        caminho = info['path']
        nome_amigavel = info['label']

        try:
            print(f"Tentando carregar {nome_amigavel}...")
            df = pd.read_parquet(caminho, engine=engine)
            dataframes[key] = df
            print(f"{nome_amigavel} carregado.")
        except Exception as e:
            print(f"Erro ao carregar {nome_amigavel}: {e}")
  
    return dataframes

if __name__ == "__main__":

    config_data = load_config()
    dfs = load_datasets(config_data)

    df_sim = dfs['sim']
    df_sinasc = dfs['sinasc']
