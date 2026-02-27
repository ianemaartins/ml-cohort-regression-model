'''
utiliza como base um país hipotético projetado para testar modelos 
de projeção populacional por coortes, baseando-se em funções que 
aproximam a evolução da natalidade, mortalidade e migração, considerando
ainda transições de ordem de parição. Estes modelos se prestam a avaliar
a aplicação da decomposição de tensores espaço-temporais em modelos
populacionais. A configuração geográfica e demográfica deste país
foi desenhada para mimetizar as tensões de fluxo e estoque 
observadas no Brasil.
'''
import os
import jax.numpy as jnp
from jax import jit
import jax.random as jrand
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly import tucker_to_tensor

tl.set_backend('jax')

class SimuladorTriadeDelta:
    def __init__(self, caminho_config="src/config_decomp.yaml"):
        # Carregamento dos parametros 
        with open(caminho_config, "r", encoding="utf-8") as f:
            self.conf = yaml.safe_load(f)

        # dimensões
        self.estados = self.conf['geografia']['todos_estados']
        self.n_est = len(self.estados)
        self.n_idade = self.conf['dimensoes_tensor']['idades']
        self.n_paridade = self.conf['dimensoes_tensor']['paridade']
        self.n_anos = self.conf['dimensoes_tensor']['anos_simulacao']

        # Pré-calculo da matemática das imputações
        self.p_espacial = self._gerar_matriz_espacial()
        self.phi_idade = self._gerar_perfil_rogers_castro()
        self.s_idade = self._gerar_curva_sobrevivencia()

        # gerar a guardar o tensor 3d
        self.p_transicao_3d = self._construir_tensor_completo()

        # inicialização dos tensores [Ano, Estado, Idade, (Paridade)]
        # O Tensor Masculino - sem paridade
        self.tensor_m = jnp.zeros((self.n_anos, self.n_est, self.n_idade))
        # O Tensor Feminino - com paridade
        self.tensor_f = jnp.zeros((self.n_anos, self.n_est, self.n_idade, self.n_paridade))

        # Tensores controle - sme migração 
        self.controle_m = jnp.zeros_like(self.tensor_m)
        self.controle_f = jnp.zeros_like(self.tensor_f)

        self._set_populacao_inicial()

    # TODO: melhorar nomeação de métodos e variáveis; melhorar docstring.
    def _gerar_matriz_espacial(self):
        """Constrói a matriz de probabilidades baseada na hierarquia regional."""
        w = np.zeros((self.n_est, self.n_est))
        regioes = self.conf['geografia']['sistemas']
        pesos = self.conf['regras_espaciais']['pesos_afinidade']
        cores = self.conf['geografia']['cores']

        # Mapeamento manual para ler o dicionário aninhado do YAML
        map_reg = {}
        for nome_grupo, dados in regioes.items():
            # Mapeia o estado principal ao grupo
            map_reg[dados['core']] = nome_grupo
            # Mapeia cada periferia ao grupo
            for p in dados['periferias']:
                map_reg[p] = nome_grupo

        for i, de in enumerate(self.estados):
            for j, para in enumerate(self.estados):
                if i == j: continue

                mesmo_g = map_reg[de] == map_reg[para]
                is_p_core = para in cores
                is_d_core = de in cores

                # Lógica de Pesos do YAML
                if mesmo_g:
                    if not is_d_core and is_p_core:
                        w[i, j] = pesos['secundario_para_core']
                    elif not is_d_core and not is_p_core:
                        w[i, j] = pesos['intra_regional']
                    else:
                        w[i, j] = pesos['retorno_core_periferia']
                else:
                    w[i, j] = pesos['inter_sistemas']

        # Normalização: Cada linha deve somar a Taxa Global de Migração
        taxa_global = self.conf['regras_espaciais']['taxa_global_migracao']
        somas = w.sum(axis=1, keepdims=True)
        p = (w / somas) * taxa_global

        # Diagonal: Quem não migra ->  1 - Taxa Global de Migração
        for i in range(self.n_est):
            p[i, i] = 1.0 - taxa_global

        return jnp.array(p)

    def _gerar_perfil_rogers_castro(self):
        """Implementa a função multi-exponencial de Rogers-Castro para intensidade por idade.
        Leva em conta compoenente infantil, adulto, idoso e constante 
        """
        x = jnp.linspace(0, self.n_idade - 1, self.n_idade)
        p = self.conf['perfil_idade_rogers_castro']

        # Componente Infantil
        c_infantil = p['intensidade_a1'] * jnp.exp(-p['queda_infantil_a1'] * x)

        # Componente Laboral (Pico dupla exponential)
        c_laboral = p['intensidade_a2'] * jnp.exp(
            -p['queda_alpha'] * (x - p['pico_idade_mu']) 
            - jnp.exp(-p['subida_lambda'] * (x - p['pico_idade_mu']))
        )

        # Componente Aposentadoria
        c_aposentadoria = p['intensidade_a3'] * jnp.exp(p['queda_aposentadoria_a2'] * (x - 65))
        c_aposentadoria = jnp.where(x < 50, 0, c_aposentadoria) # Só ativa após a idade determinada

        phi = c_infantil + c_laboral + c_aposentadoria

        # Normaliza para média 1.0 para não alterar a massa total da taxa global
        return phi / jnp.mean(phi)

    def _gerar_curva_sobrevivencia(self):
        """Gera probabilidade de sobrevivência por idade via curva de Gompertz.
        s(x) = exp(-alpha * exp(beta * x))
        """
        x = jnp.arange(self.n_idade, dtype=jnp.float32)
        g = self.conf['regra_temporal']
        mu = g['gompertz_alpha'] * jnp.exp(g['gompertz_beta'] * x)
        return jnp.exp(-mu)

    def _set_populacao_inicial(self):
        """Distribui a população de 1990 seguindo uma pirâmide etária jovem."""

        idades = jnp.linspace(0, 5, self.n_idade)
        base = jnp.exp(-0.35 * idades) # Pirâmide brasil 1990
        plateau = jnp.where(idades < 1.5, 1.0, jnp.exp(-0.5 * (idades - 1.5)))
        dist_idade = base * plateau
        dist_idade /= dist_idade.sum()

        for i, est in enumerate(self.estados):
            pop_total = self.conf['populacao_inicial_1990'][est]

            # Homens (M)
            self.tensor_m = self.tensor_m.at[0, i, :].set(dist_idade * pop_total * 0.49)

            # Mulheres (F) - Inicialmente com paridade 0
            self.tensor_f = self.tensor_f.at[0, i, :, 0].set(dist_idade * pop_total * 0.51)

        # Controle começa igual à população inicial
        self.controle_m = self.controle_m.at[0].set(self.tensor_m[0])
        self.controle_f = self.controle_f.at[0].set(self.tensor_f[0])

    def _construir_tensor_completo(self):
        """Monta o tensor [De, Para, Idade] uma única vez para uso global."""
        # Pegamos as regras espaciais
        idx_cores = [self.estados.index(c) for c in self.conf['geografia']['cores']]
        taxa_mig_global = self.conf['regras_espaciais']['taxa_global_migracao']

        # Propensão de Saída
        propensao_saida = jnp.full((self.n_est,), taxa_mig_global)
        propensao_saida = propensao_saida.at[jnp.array(idx_cores)].set(taxa_mig_global * 0.2)

        # Matriz de Destinos (Atração)
        p_destinos = self.p_espacial * (jnp.ones((self.n_est, self.n_est)) - jnp.eye(self.n_est))
        p_destinos = p_destinos.at[:, jnp.array(idx_cores)].multiply(5.0)
        p_destinos = p_destinos / jnp.sum(p_destinos, axis=1, keepdims=True)

        # Intensidade por Idade
        fator_idade = self.phi_idade / jnp.mean(self.phi_idade)

        # Montagem do Tensor 3D inicial
        p_3d = (p_destinos[:, :, jnp.newaxis] * propensao_saida[
            :, jnp.newaxis, jnp.newaxis] * fator_idade[jnp.newaxis, jnp.newaxis, :])

        # Garantia de Conservação (Diagonal)
        soma_saidas = jnp.sum(p_3d, axis=1)
        soma_saidas = jnp.minimum(soma_saidas, 0.95)

        for k in range(self.n_idade):
            diag_k = 1.0 - soma_saidas[:, k]
            p_3d = p_3d.at[jnp.arange(self.n_est), jnp.arange(self.n_est), k].set(diag_k)
   
        return p_3d

    # O (0, 3, 4) indica que self, aplicar_migracao e t não devem ser
    # transformados em arrays do JAX.
    @partial(jit, static_argnums=(0, 3, 4))
    def _evoluir_ano(self, pop_m, pop_f, aplicar_migracao=True, t=0):
        """
        Evolui a população masculina e feminina em um passo anual, incorporando
        envelhecimento, mortalidade, fecundidade e migração espacial.

        Esta função implementa a dinâmica demográfica básica do modelo, realizando:
        (i) o envelhecimento das coortes etárias com aplicação de uma probabilidade
        de sobrevivência constante; (ii) o cálculo de nascimentos a partir da
        população feminina em idade fértil; (iii) a inserção da coorte etária zero
        (recém-nascidos), respeitando a razão de sexo ao nascimento; e
        (iv) opcionalmente, a redistribuição espacial da população via migração,
        preservando o total populacional.

        Args: 
            pop_m : jax.numpy.ndarray
                Tensor da população masculina no ano corrente, com dimensões
                (n_regiões, n_idades).

            pop_f : jax.numpy.ndarray
                Tensor da população feminina no ano corrente, com dimensões
                (n_regiões, n_idades, n_coortes ou subgrupos adicionais).

            aplicar_migracao : bool, opcional
                Indica se a migração espacial deve ser aplicada neste passo temporal.
                Quando False, apenas a dinâmica demográfica interna é considerada.
                O padrão é True.

        Returns:
            prox_m : jax.numpy.ndarray
                População masculina no ano seguinte, após envelhecimento,
                mortalidade, nascimentos e migração (se aplicada).

            prox_f : jax.numpy.ndarray
                População feminina no ano seguinte, após envelhecimento,
                mortalidade, nascimentos e migração (se aplicada).

        """
        # parâmetros
        rsn = self.conf['regras_espaciais']['razao_sexo_nascimento']
        f_ini = self.conf['regra_temporal']['fecundidade_inicio']
        f_fim = self.conf['regra_temporal']['fecundidade_fim']
        taxa_fec = f_ini * (f_fim / f_ini) ** (t / (self.n_anos - 1))

        # --- FECUNDIDADE E TRANSIÇÃO DE PARIDADE
        idades = jnp.arange(self.n_idade)
        fec_mask = (idades >= 15) & (idades <= 49)
        pi_fec = jnp.where(fec_mask, taxa_fec, 0.0)

        sobreviventes_f = pop_f * self.s_idade[None, :, None]

        # Mulheres que NÃO estão na paridade máxima: podem avançar
        novas_maes_transicao = sobreviventes_f[:, :, :-1] * pi_fec[None, :, None]
        # Mulheres na paridade máxima: têm filhos mas permanecem no mesmo índice
        novas_maes_teto = sobreviventes_f[:, :, -1:] * pi_fec[None, :, None]

        # Envelhecimento + Transição
        prox_f = jnp.zeros_like(pop_f)

        # Aquelas que NÃO tiveram filhos permanecem na mesma paridade e envelhecem
        permanecem_na_ordem = sobreviventes_f.at[:, :, :-1].multiply(1.0 - pi_fec[None, :, None])
        # Paridade máxima: sobreviventes que não tiveram filhos + as que tiveram (ficam no teto)
        permanecem_na_ordem = permanecem_na_ordem.at[:, :, -1].set(
            sobreviventes_f[:, :, -1] * (1.0 - pi_fec[None, :]) + novas_maes_teto[:, :, 0]
        )
        prox_f = prox_f.at[:, 1:, :].add(permanecem_na_ordem[:, :-1, :])

        # Aquelas que tiveram filhos: avançam para paridade 'o+1' e envelhecem
        prox_f = prox_f.at[:, 1:, 1:].add(novas_maes_transicao[:, :-1, :])

        # --- NASCIMENTOS
        total_nascidos = jnp.sum(novas_maes_transicao, axis=(1, 2)) + jnp.sum(novas_maes_teto, axis=(1, 2))

        # Evolução masculina
        prox_m = jnp.zeros_like(pop_m)
        prox_m = prox_m.at[:, 1:].set(pop_m[:, :-1] * self.s_idade[None, :-1])
        
        # Inserção da Coorte Zero em ambos os sexos
        prox_m = prox_m.at[:, 0].set(total_nascidos * (rsn / (1 + rsn)))
        prox_f = prox_f.at[:, 0, 0].set(total_nascidos * (1 / (1 + rsn)))

        # --- MIGRAÇÃO COM CONSERVAÇÃO
        if aplicar_migracao:
            prox_m = jnp.einsum('ik,ijk->jk', prox_m, self.p_transicao_3d)
            prox_f = jnp.einsum('ikp,ijk->jkp', prox_f, self.p_transicao_3d)

        return prox_m, prox_f

    def simular(self):
        """Roda o loop de anos para População Real e População de Controle."""
        f_ini = self.conf['regra_temporal']['fecundidade_inicio']
        f_fim = self.conf['regra_temporal']['fecundidade_fim']
        ordens = np.arange(self.n_paridade)

        print(f"{'Ano':>6} {'Taxa Fec':>10} {'Filhos Medios':>14}")
        print("-" * 34)

        for t in range(self.n_anos - 1):
            taxa_fec = f_ini * (f_fim / f_ini) ** (t / (self.n_anos - 2))

            # Evolução Real (Com Migração)
            m_real, f_real = self._evoluir_ano(self.tensor_m[t], self.tensor_f[t], True, t)
            self.tensor_m = self.tensor_m.at[t+1].set(m_real)
            self.tensor_f = self.tensor_f.at[t+1].set(f_real)

            # Evolução Controle (Sem Migração)
            m_ctrl, f_ctrl = self._evoluir_ano(self.controle_m[t], self.controle_f[t], False, t)
            self.controle_m = self.controle_m.at[t+1].set(m_ctrl)
            self.controle_f = self.controle_f.at[t+1].set(f_ctrl)

            # Métrica: número médio de filhos ponderado pela distribuição de paridade
            dist_par = np.array(self.tensor_f[t+1].sum(axis=(0, 1)))  # [Paridade]
            total_f = dist_par.sum()
            filhos_medios = float((dist_par * ordens).sum() / total_f) if total_f > 0 else 0.0

            ano = self.conf['dimensoes_tensor'].get('anos_simulacao_inicio', 1990) + t
            print(f"{ano:>6} {taxa_fec:>10.4f} {filhos_medios:>14.4f}")

        print("\nSimulação finalizada.")

    def calcular_residuo(self):
        """Retorna o Tensor R = Pop_Real - Pop_Controle."""
        # Somamos a paridade no feminino para ter [Ano, Estado, Idade]

        res_m = self.tensor_m - self.controle_m
        res_f = jnp.sum(self.tensor_f, axis=-1) - jnp.sum(self.controle_f, axis=-1)

        # Retornamos o resíduo total (M+F) para a decomposição
        return res_m + res_f

    def aplicar_ruido(self, residuo, intensidade=0.05, df=5):
        """Aplica ruído t de Student (df graus de liberdade) antes da decomposição."""
        chave_z, chave_chi2 = jrand.split(jrand.PRNGKey(42))

        z = jrand.normal(chave_z, residuo.shape)
        chi2 = 2.0 * jrand.gamma(chave_chi2, df / 2.0, shape=residuo.shape)
        ruido = (z / jnp.sqrt(chi2 / df)) * intensidade * jnp.abs(residuo)

        return residuo + ruido

    def decompor_padroes(self, residuo_com_ruido, ranks=(5, 3, 10)):
        """Decompõe o resíduo via Tucker: núcleo + fatores por dimensão [Tempo, Espaço, Idade]."""
        tl.set_backend('jax')

        # Normalização por estado para equilibrar estados de diferentes tamanhos
        std = jnp.std(residuo_com_ruido, axis=(0, 2), keepdims=True) + 1e-8
        residuo_norm = residuo_com_ruido / std

        nucleo, fatores = tucker(residuo_norm, rank=ranks)

        # Erro relativo de reconstrução
        reconstruido = tucker_to_tensor((nucleo, fatores))
        erro = tl.norm(residuo_norm - reconstruido) / tl.norm(residuo_norm)
        print(f"Erro relativo de reconstrução Tucker: {float(erro):.4f}")

        return nucleo, fatores, std


class VisualizadorTriadeDelta:
    def __init__(self, simulador, dados_custom=None, output_path="resultados_graficos"):
        """
        Classe para visualização dos resultados da simulação.
        :param simulador: Instância da classe SimuladorTriadeDelta já executada.
        :param output_path: Pasta onde os gráficos serão salvos.
        """
        self.sim = simulador
        self.estados = simulador.estados
        
        # Lógica de anos (simplificada)
        ano_ini = simulador.conf['dimensoes_tensor'].get('anos_simulacao_inicio', 1990)
        self.anos = np.arange(ano_ini, ano_ini + simulador.n_anos)

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        if dados_custom is not None:
            # Se passarmos o resíduo, tratamos ele como a "população" a ser plotada
            # Dividimos por 2 apenas para manter a compatibilidade com funções que somam M + F
            self.m = np.array(dados_custom) / 2
            self.f = np.array(dados_custom) / 2
        else:
            # Caso contrário, usa os dados normais da simulação
            self.m = np.array(simulador.tensor_m)
            self.f = np.array(jnp.sum(simulador.tensor_f, axis=-1))

    def _agrupar_quinquenal(self, dados_idade):
        """Agrupa dados de idades simples (0-90) em faixas quinquenais."""
        agrupado = []
        for i in range(0, 90, 5):
            agrupado.append(dados_idade[i:i+5].sum())
        agrupado.append(dados_idade[90:].sum()) 
        return np.array(agrupado)

    def plot_piramide(self, ano_alvo, estado_nome):
        """Gera e salva a Pirâmide Etária (Homens vs Mulheres)."""
        idx_t = ano_alvo - self.anos[0]
        idx_est = self.estados.index(estado_nome)

        m_quinq = self._agrupar_quinquenal(self.m[idx_t, idx_est])
        f_quinq = self._agrupar_quinquenal(self.f[idx_t, idx_est])

        faixas = [f"{i}-{i+4}" for i in range(0, 90, 5)] + ["90+"]

        plt.figure(figsize=(10, 7))
        plt.barh(faixas, -m_quinq, color='royalblue', label='Homens', alpha=0.8)
        plt.barh(faixas, f_quinq, color='lightpink', label='Mulheres', alpha=0.8)

        plt.axvline(0, color='black', lw=1)
        plt.title(f"Pirâmide Etária - Estado {estado_nome} ({ano_alvo})")
        plt.xlabel("População")
        plt.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.4)

        # Salvamento
        filename = f"piramide_{estado_nome}_{ano_alvo}.png"
        plt.savefig(os.path.join(self.output_path, filename), dpi=300, bbox_inches='tight')
        plt.close() # Fecha a figura para liberar memória
        print(f"Gráfico salvo: {filename}")

    def plot_evolucao_temporal(self, estados_nomes):
        """Compara a evolução da população total entre estados selecionados."""
        plt.figure(figsize=(12, 6))

        for nome in estados_nomes:
            idx = self.estados.index(nome)
            pop_total = self.m[:, idx, :].sum(axis=1) + self.f[:, idx, :].sum(axis=1)
            plt.plot(self.anos, pop_total, marker='s', markersize=4, label=f"Estado {nome}")

        plt.title("Crescimento Populacional Acumulado por Unidade Federativa")
        plt.xlabel("Ano")
        plt.ylabel("Habitantes")
        plt.legend()
        plt.grid(True, alpha=0.3)

        filename = "evolucao_temporal_comparativa.png"
        plt.savefig(os.path.join(self.output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo: {filename}")

    def plot_variacao_etaria(self, nome_estado):
        """Barras mostrando o crescimento/decréscimo por faixa etária entre 1990 e o fim."""
        idx = self.estados.index(nome_estado)

        pop_inicial = self._agrupar_quinquenal(self.m[0, idx] + self.f[0, idx])
        pop_final = self._agrupar_quinquenal(self.m[-1, idx] + self.f[-1, idx])
        variacao = pop_final - pop_inicial

        faixas = [f"{i}-{i+4}" for i in range(0, 90, 5)] + ["90+"]
        cores = ['seagreen' if v > 0 else 'indianred' for v in variacao]

        plt.figure(figsize=(12, 6))
        plt.bar(faixas, variacao, color=cores, alpha=0.8)
        plt.xticks(rotation=45)
        plt.title(f"Variação Líquida da Estrutura Etária: {nome_estado} (Final vs Inicial)")
        plt.ylabel("Diferença Absoluta de Habitantes")
        plt.axhline(0, color='black', lw=1)

        filename = f"variacao_etaria_{nome_estado}.png"
        plt.savefig(os.path.join(self.output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo: {filename}")

    def plot_assinatura_migratoria(self):
        """Heatmap do Resíduo Total: Revela os polos de atração e expulsão."""
        # Cálculo do Resíduo dentro da visualização (Real - Controle)
        residuo_m = self.m - np.array(self.sim.controle_m)
        residuo_f = self.f - np.array(jnp.sum(self.sim.controle_f, axis=-1))
        residuo_total = residuo_m + residuo_f

        # Agregamos por Estado e Ano (Soma de todas as idades)
        matriz_heatmap = residuo_total.sum(axis=2).T 

        plt.figure(figsize=(14, 8))
        sns.heatmap(matriz_heatmap, xticklabels=self.anos, yticklabels=self.estados, 
                    cmap="RdBu", center=0, cbar_kws={'label': 'Saldo Migratório Líquido'})

        plt.title("Assinatura Migratória no Tempo e Espaço (Resíduo Líquido)")
        plt.xlabel("Ano")
        plt.ylabel("Estado")

        filename = "heatmap_assinatura_migratoria.png"
        plt.savefig(os.path.join(self.output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico salvo: {filename}")

    def plot_fluxos_brutos(self, ano_alvo, estado_nome):
        """
        Gera um gráfico de barras comparando Imigração (quem entra) 
        e Emigração (quem sai) por faixa etária.
        """
        idx_t = ano_alvo - self.anos[0]
        idx_est = self.estados.index(estado_nome)

        # Recupera a população do estado no ano alvo
        pop_m = self.m[idx_t] # [Estado, Idade]
        pop_f = self.f[idx_t] # [Estado, Idade]
        pop_total_est = pop_m + pop_f # [Estado, Idade]

        # Recria a Matriz de Transição 3D para esse cálculo
        p_comp = np.array(self.sim.p_transicao_3d) # [De, Para, Idade]

        # Cálculo de quem sai
        prob_sair = 1.0 - np.diagonal(p_comp, axis1=0, axis2=1) # [Idade, Estado]
        emigrantes = pop_total_est[idx_est, :] * prob_sair[:, idx_est]

        # Cálculo de quem chega
        imigrantes = np.zeros(self.sim.n_idade)
        for i in range(self.sim.n_est):
            if i == idx_est: continue
            # Pessoas que saem do estado 'i' para o nosso estado alvo
            imigrantes += pop_total_est[i, :] * p_comp[i, idx_est, :]

        # Agrupamento Quinquenal para o gráfico
        in_quinq = self._agrupar_quinquenal(imigrantes)
        out_quinq = self._agrupar_quinquenal(emigrantes)

        faixas = [f"{i}-{i+4}" for i in range(0, 90, 5)] + ["90+"]

        # Plotagem
        x = np.arange(len(faixas))
        largura = 0.4

        plt.figure(figsize=(14, 7))
        plt.bar(x - largura/2, in_quinq, largura, label='Chegaram (Imigração)', color='teal', alpha=0.8)
        plt.bar(x + largura/2, out_quinq, largura, label='Saíram (Emigração)', color='coral', alpha=0.8)

        plt.title(f"Fluxos Migratórios Brutos por Idade - Estado {estado_nome} ({ano_alvo})")
        plt.xticks(x, faixas, rotation=45)
        plt.ylabel("Número de Pessoas")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        filename = f"fluxos_brutos_{estado_nome}_{ano_alvo}.png"
        plt.savefig(os.path.join(self.output_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de fluxos brutos salvo: {filename}")



if __name__ == "__main__":

    sim = SimuladorTriadeDelta()
    sim.simular()
    residuo_tensor = sim.calcular_residuo()
    residuo_tensor_ruidoso = sim.aplicar_ruido(residuo_tensor, intensidade=0.9)
    pesos_migracao, fatores_migracao = sim.decompor_padroes(residuo_tensor_ruidoso, rank=3)
    print(f"Tensor de Resíduo Migratório gerado: {residuo_tensor_ruidoso.shape}")

    #visualização
    viz = VisualizadorTriadeDelta(sim, residuo_tensor_ruidoso, output_path="graficos_artigo")

    # Gerar pirâmide da região x no ano y
    viz.plot_piramide(2000, "A")
    # Comparar 3 estados
    viz.plot_evolucao_temporal(["A", "A1", "A2"])
    # Ver a variação de um estado
    viz.plot_variacao_etaria("A")
    # Gerar o mapa de calor completo
    viz.plot_assinatura_migratoria()
    # Fluxo bruo de migração do estado x no ano y com todas as idades
    viz.plot_fluxos_brutos(2010, 'A')
