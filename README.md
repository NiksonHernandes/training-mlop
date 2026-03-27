# Definição do Problema — Cardiotocografia Fetal

**Autor:** Renan Santos Mendes  
**Instituição:** PUC Minas Virtual

---

## 📋 Visão Geral

Este projeto aborda o problema do **sofrimento fetal** e o uso do exame de
**cardiotocografia** como ferramenta de diagnóstico, com foco na classificação
automatizada dos resultados por meio de modelos de machine learning.

---

## 🏥 O que é Cardiotocografia?

A cardiotocografia (CTG) é um exame simples e de baixo custo utilizado no
monitoramento fetal. Seu principal objetivo é a **prevenção da mortalidade
infantil e materna**.

O equipamento funciona enviando pulsos ultrassônicos e lendo sua resposta,
capturando os seguintes sinais:

- **Frequência cardíaca fetal (FHR)**
- **Movimentos fetais**
- **Contrações uterinas (UC)**

---

## 🔄 Fluxo de Trabalho
```
Dado/Exame  ──►  Extração de Features  ──►  Modelo de Classificação
```

### Features Extraídas

| Feature | Descrição |
|---|---|
| Frequência | Frequência cardíaca fetal |
| Movimentos | Movimentos do feto |
| Contrações | Contrações uterinas |
| Acelerações | Acelerações da frequência cardíaca |
| Desacelerações | Desacelerações da frequência cardíaca |

---

## 🎯 Classes de Saída

Os dados são classificados em **3 categorias**:

| Classe | Descrição |
|---|---|
| ✅ Normal | Estado fetal saudável |
| ⚠️ Suspeito | Requer atenção e acompanhamento |
| 🚨 Patológico (Doente) | Indica sofrimento fetal |

---

## 📚 Referências

- PUC Minas Virtual — Material de curso