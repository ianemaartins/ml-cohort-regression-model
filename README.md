## Modelo de projeção populacional por coortes

O presente repositório foi destinado para o desenvolvimento e documentação de um modelo de projeção populacional por coortes proposto como avaliação da disciplina de Matemática para Ciência de Dados na UFES. O modelo em questão leva em consideração questões de natalidade, mortalidade e migração populacional, tendo o objetivo de avaliar a aplicação de decomposição de tensores espaço-temporais em modelos populacionais. 

## Etapas do projeto:

### 1. Dados e bibliotecas necessárias:

Para a coleta de dados foi utilizada a biblioteca microdatasus. Apesar de haver preferência pela linguagem Python, a biblioteca em questão, desenvolvida em R, é muito eficiente na coleta de dados governamentais brasileiros e por esse motivo foi utilizada. Para além disso, foi criado um ambiente virtual e instaladas as bibliotecas jax, jaxlib, xarray, tensorly, microdatasus, sidrapy.

- Instalar ambiente virtual
- Instalar bibliotecas:  jax, jaxlib, xarray, tensorly, microdatasus, sidrapy
- Coletar dados (etapa realizada em R devido a existência da biblioteca microdatasus)

### 2. Prova de conceito

Como não existem dados que explicitem de fato a migração no Brasil, propois-se realizar uma prova de conceito para entender quais os elementos que explicam melhor os fluxos migratórios. A ideia principal é a de que  fosse feita uma investigação do caráter migratório em um ambiente controlado, com dados fictícios e que, posteriormente, as descobertas fossem expandidas para o modelo com dados reais. 

