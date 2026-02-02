install.packages("remotes")
remotes::install_github("rfsaldanha/microdatasus", auth_token = NULL)

library(microdatasus)
library(arrow)
library(dplyr)

# ------- FUNÇÃO DE EXTRAÇÃO DE DADOS

extrair_datasus_parquet <- function(sistema, anos, ufs, colunas_selecionadas, pasta_destino) {
  
  # Cria a pasta de destino se não existir
  if (!dir.exists(pasta_destino)) dir.create(pasta_destino, recursive = TRUE)
  
  for (ano in anos) {
    arquivo_nome <- file.path(pasta_destino, paste0(sistema, "_", ano, ".parquet"))
    
    # Checkpoint: Pula se já existir
    if (file.exists(arquivo_nome)) {
      message(paste(">", sistema, ano, "já existe. Pulando..."))
      next
    }
    
    message(paste("--- Baixando", sistema, "ano:", ano, "---"))
    
    try({
      # Download
      df <- fetch_datasus(year_start = ano, year_end = ano, 
                          uf = ufs, information_system = sistema)
      
      # Processamento específico
      if (sistema == "SIM-DO") {
        df <- process_sim(df)
      } else if (sistema == "SINASC") {
        df <- process_sinasc(df)
      }
      
      # Seleção de colunas e controle
      df_resumo <- df %>%
        select(any_of(colunas_selecionadas)) %>%
        mutate(ANO_REF = ano)
      
      # Salvamento
      write_parquet(df_resumo, arquivo_nome)
      message(paste("Sucesso:", sistema, ano, "salvo."))
      
      # Limpeza de memória
      rm(df, df_resumo)
      gc()
      
    }, silent = FALSE)
  }
  
  # Consolidação Final
  message(paste("Consolidando arquivos de", sistema, "..."))
  dataset <- open_dataset(pasta_destino)
  nome_final <- paste0(sistema, "_1990_2022_CONSOLIDADO.parquet")
  write_parquet(dataset, nome_final)
  
  return(nome_final)
}

# --- PARÂMETROS GERAIS
meus_anos <- 1990:2022
minhas_ufs <- c("AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", 
                "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", 
                "SP", "SE", "TO")

# --- EXECUÇÃO PARA O SIM (ÓBITOS) ---
colunas_sim <- c("ano_obito", "ano_nasc", "def_sexo", "CONTADOR", "NATURAL", "IDADE", "SEXO")

extrair_datasus_parquet(
  sistema = "SIM-DO",
  anos = meus_anos,
  ufs = minhas_ufs,
  colunas_selecionadas = colunas_sim,
  pasta_destino = "dados_anuais_sim"
)

# --- EXECUÇÃO PARA O SINASC (NASCIMENTOS) ---
# Adicionei as colunas que você pediu: idade da mãe e número de filhos anteriores
colunas_sinasc <- c("DTNASC", "SIGLA_UF", "IDADEMAE", "QTDFILANT", "QTDFILVIVO", "QTDFILMORT", "SEXO")

extrair_datasus_parquet(
  sistema = "SINASC",
  anos = meus_anos,
  ufs = minhas_ufs,
  colunas_selecionadas = colunas_sinasc,
  pasta_destino = "dados_anuais_sinasc"
)
