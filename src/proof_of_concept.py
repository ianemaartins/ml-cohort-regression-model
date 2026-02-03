'''
utiliza como base um país hipotético projetado para testar modelos 
de projeção populacional por coortes, baseando-se em funções que 
aproximam a evolução da natalidade, mortalidade e migração, considerando
ainda transições de ordem de parição. Estes modelos se prestam a avaliar
a aplicação da decomposição de tensores espaço-temporais em modelos
populacionais. A configuração geográfica e demográfica deste país
foi desenhada para mimetizar as tensões de fluxo e estoque 
observadas em grandes nações em desenvolvimento, como o Brasil.
'''
import os
from typing import Tuple
import jax.numpy as jnp
from jax import jit
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from tensorly.cp_tensor import cp_to_tensor
import matplotlib.pyplot as plt
import seaborn as sns

# Configura o backend para JAX para suportar diferenciação e GPU/TPU se disponível
tl.set_backend('jax')

@jit
def project_female_population(
    current_pop: jnp.ndarray, 
    survival_rate: float, 
    fertility_rate: float, 
    net_migration: jnp.ndarray
) -> jnp.ndarray:
    """
    Projeta a população feminina para o próximo ano considerando envelhecimento,
    transições de paridade (partos) e migração.

    Args:
        current_pop: Tensor da população (região, idade, paridade).
        survival_rate: Probabilidade de sobrevivência para a próxima idade.
        fertility_rate: Probabilidade de transição de paridade (ter um filho).
        net_migration: Fluxo líquido de pessoas para cada célula do tensor.

    Returns:
        jnp.ndarray: População feminina projetada para t+1.
    """
    # Sobreviventes que permanecem na mesma paridade (não tiveram filhos este ano)
    stay_in_parity = current_pop * survival_rate * (1 - fertility_rate)

    # Transição de Paridade: Mães que mudaram da paridade 'o' para 'o+1'
    # O deslocamento ocorre no último eixo (paridade)
    new_mothers = current_pop * survival_rate * fertility_rate
    shifted_parity = jnp.pad(new_mothers[:, :, :-1], ((0, 0), (0, 0), (1, 0)))

    # População no próximo ano = Permanência + Transições + Migração
    return stay_in_parity + shifted_parity + net_migration

@jit
def project_male_population(
    current_pop: jnp.ndarray, 
    survival_rate: float, 
    net_migration: jnp.ndarray
) -> jnp.ndarray:
    """
    Projeta a população masculina baseada em sobrevivência e migração.

    Args:
        current_pop: Tensor da população (região, idade).
        survival_rate: Probabilidade de sobrevivência para a próxima idade.
        net_migration: Fluxo líquido de pessoas para cada célula.

    Returns:
        jnp.ndarray: População masculina projetada para t+1.
    """
    return (current_pop * survival_rate) + net_migration

def generate_synthetic_migration_tensor(
    years: int = 30, 
    regions: int = 12, 
    cohorts: int = 5
) -> jnp.ndarray:
    """
    Cria um tensor de resíduos sintético (R) com padrões migratórios embutidos.

    Args:
        years: Número de anos na série temporal.
        regions: Número de regiões geográficas.
        cohorts: Número de faixas etárias (coortes).

    Returns:
        jnp.ndarray: Tensor de migração (regiões, anos, coortes).
    """
    shape = (regions, years, cohorts)
    residual_tensor = jnp.zeros(shape)

    # Comunidade 1: Regiões 0-3 atraindo jovens (índice 2)
    residual_tensor = residual_tensor.at[0:4, :, 2].add(10.0)

    # Comunidade 2: Regiões 4-7 atraindo adultos (índice 3)
    residual_tensor = residual_tensor.at[4:8, :, 3].add(15.0)

    return jnp.abs(residual_tensor)

def run_tensor_decomposition(
    residual_tensor: jnp.ndarray, 
    n_components: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Decompõe o tensor em matrizes de fatores: Espacial, Temporal e Coorte.

    Args:
        residual_tensor: O tensor de entrada (local x ano x idade).
        n_components: Rank da decomposição (número de padrões).

    Returns:
        Tuple: Matrizes de fatores (spatial, temporal, cohort).
    """
    # NTF (Non-negative Tensor Factorization) para manter interpretabilidade demográfica
    weights, factors = non_negative_parafac(
        residual_tensor,
        rank=n_components,
        init='random',
        n_iter_max=500
    )
    
    return factors[0], factors[1], factors[2]

def calculate_reconstruction_error(
    original_tensor: jnp.ndarray, 
    factors: list[jnp.ndarray]
) -> float:
    """
    Calcula o erro relativo entre o tensor original e sua reconstrução via CP.

    Args:
        original_tensor: O tensor residual real.
        factors: lista de matrizes de fatores [Espacial, Temporal, Coorte].

    Returns:
        float: Erro relativo (quanto mais próximo de 0, melhor).
    """
    # Reconstroi o tensor a partir dos fatores
    reconstructed_tensor = cp_to_tensor((None, factors))
    
    error_norm = jnp.linalg.norm(original_tensor - reconstructed_tensor)
    original_norm = jnp.linalg.norm(original_tensor)
    
    return float(error_norm / original_norm)

def plot_spatial_factors(spatial_factor: jnp.ndarray, output_dir: str) -> None:
    """Gera um heatmap dos fatores espaciais."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    sns.heatmap(spatial_factor, annot=True, cmap="YlGnBu", 
                yticklabels=[f"Região {i}" for i in range(spatial_factor.shape[0])],
                xticklabels=[f"Comp {i+1}" for i in range(spatial_factor.shape[1])])
    plt.title("Fatores Espaciais: Identificação de Comunidades Migratórias")
    plt.savefig(os.path.join(output_dir, "spatial_factors.png"), dpi=300)
    plt.close()

def plot_temporal_factors(temporal_factor: jnp.ndarray, output_dir: str) -> None:
    """Gera gráfico de linhas da evolução temporal dos fatores."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 5))
    for i in range(temporal_factor.shape[1]):
        plt.plot(temporal_factor[:, i], label=f"Componente {i+1}", marker='o', markersize=4)
    
    plt.title("Fatores Temporais: Evolução dos Fluxos Migratórios")
    plt.xlabel("Anos")
    plt.ylabel("Intensidade")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "temporal_factors.png"), dpi=300)
    plt.close()

def plot_cohort_factors(cohort_factor: jnp.ndarray, output_dir: str) -> None:
    """Gera gráfico de barras para o perfil etário das coortes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 5))
    x = jnp.arange(cohort_factor.shape[0])
    width = 0.2
    
    for i in range(cohort_factor.shape[1]):
        plt.bar(x + i*width, cohort_factor[:, i], width, label=f"Componente {i+1}")

    plt.title("Fatores de Coorte: Perfil de Idade dos Migrantes")
    plt.xticks(x + width, [f"Idade {i}" for i in range(cohort_factor.shape[0])])
    plt.legend()
    plt.savefig(os.path.join(output_dir, "cohort_factors.png"), dpi=300)
    plt.close()

def main() -> None:
    folder = "plots"
    print("--- Iniciando Simulação com Tensores (JAX + TensorLy) ---")

    # Parâmetros: 12 regiões, 5 idades, 3 níveis de paridade
    n_regions, n_ages, n_parities = 12, 5, 3
    survival_rate = 0.95
    fertility_rate = 0.2 # Valor mais realista para probabilidade anual de parto

    # Inicialização da população
    pop_female_t = jnp.ones((n_regions, n_ages, n_parities)) * 100
    pop_male_t = jnp.ones((n_regions, n_ages)) * 100

    # Geração de migração sintética para 1 ano (ajuste de shape para projeção)
    true_migration = generate_synthetic_migration_tensor(years=1, regions=n_regions, cohorts=n_ages)
    # Expande dimensão para bater com a paridade feminina (repetindo a migração entre paridades)
    female_migration = jnp.expand_dims(true_migration[:, 0, :], axis=-1) / n_parities

    # Execução da Projeção
    pop_female_next = project_female_population(pop_female_t, survival_rate, fertility_rate, female_migration)
    pop_male_next = project_male_population(pop_male_t, survival_rate, true_migration[:, 0, :])

    # Decomposição de Tensor de Longo Prazo (30 anos)
    print("Gerando Tensor de Resíduos (30 anos) e executando NTF...")
    residual_tensor_30y = generate_synthetic_migration_tensor(years=30)
    spatial, temporal, cohort = run_tensor_decomposition(residual_tensor_30y, n_components=3)

    # Validação e Visualização
    error = calculate_reconstruction_error(residual_tensor_30y, [spatial, temporal, cohort])
    print(f"Erro de Reconstrução: {error:.8f}")
    
    plot_spatial_factors(spatial, folder)
    plot_temporal_factors(temporal, folder)
    plot_cohort_factors(cohort, folder)
    print(f"Processo finalizado. Gráficos salvos em: {folder}/")

if __name__ == "__main__":
    main()