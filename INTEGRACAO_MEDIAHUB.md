# Integrar o CodecClean no Media Hub

Guia de implementação do filtro de resíduo de compressão que agora roda
**dentro** do worker. Escrito em 2026-08-19, contra o worker no commit
`8255d70` do fork.

---

## Resposta curta: não refaça, troque o binário

O código de integração do Media Hub (`src-tauri/src/rtx.rs`) **está bom**.
Ele monta os argumentos, registra o processo filho para o cancelamento
funcionar, quebra o stderr em `\r`/`\n` para ler a barra de progresso,
guarda as linhas que não são progresso como cauda de erro, e distingue
cancelamento de crash na ordem certa — com o comentário explicando por
quê. Isso não é código de teste; é código que alguém acertou com esforço.

**O que está desatualizado é o binário**, que é de 2026-07-05 e não tem
nenhum fix do merge `d27a9bb` de 18/07:

| defeito | consequência |
|---|---|
| OOG hard-clamp no round-trip RGB | cores fora de gama destruídas (235/180/180 → 211/141/144) |
| quadro 0 duplicado | vídeo inteiro deslocado em 1 quadro |
| último quadro perdido | duração quase certa, e ninguém nota |

Refazer do zero jogaria fora o encanamento bom para reescrever o que já
funciona. **A mudança é cirúrgica: trocar o binário e somar três
argumentos.**

---

## O que muda, em ordem

### 1. Publicar o binário novo

O worker é um **sidecar baixado sob demanda** para `<app_data>/bin/`
(`worker_path()` em `rtx.rs:242`), não algo empacotado no app. Então
atualizar é **publicar um binário novo**, não reempacotar o Media Hub —
e quem já tem o velho pega o novo pelo mesmo caminho de instalação de
ferramenta que o `tools.rs` usa (`ToolSpec` → baixa zip → extrai →
`<app_data>/bin`).

Binário: `F:\CLAUDE\rtx-worker-fork\build-mod\RTXVideoProcessor.exe`
(30,4 MB). Vai junto, como já vai hoje: `nvngx_vsr.dll` e
`nvngx_truehdr.dll`.

### 2. Empacotar os pesos

Arquivo: `cc_32x4.blob`, **310 KB**. Mesma rota do binário, mesmo
diretório (`<app_data>/bin/cc_32x4.blob`). É o modelo inteiro — não há
mais nada para embarcar, sem PyTorch, sem CUDA toolkit, sem TensorRT.

> O blob é um contrato de ORDEM: cabeçalho `CCNET\0\0\0` + `ch`,
> `blocks`, `cond` + os pesos na ordem que o `eval/dump_blob.py` grava.
> Trocar o modelo = gerar blob novo com aquele script, não editar bytes.

### 3. Somar os argumentos

Em `rtx.rs`, onde hoje se monta o `args` (linha ~707):

```rust
// filtro de resíduo de compressão: só quando o usuário liga
if cc_enabled {
    args.push("--cc-blob".into());
    args.push(blob_path.to_string_lossy().to_string());
    args.push("--cc-strength".into());
    args.push(format!("{:.2}", cc_strength));   // 0.00 a 1.00
}
// preset de qualidade da saída
args.push("--quality".into());
args.push(quality.into());   // lossless | master | entrega | previa
```

**Se o blob não carregar, o worker sai com erro** — de propósito. O
usuário pediu o filtro; seguir sem ele entregaria um vídeo
silenciosamente diferente do pedido.

### 4. UI

| controle | faixa | observação |
|---|---|---|
| ligar/desligar o filtro | — | desligado por padrão |
| força (`--cc-strength`) | 0,00 a 1,00 | **1,00 é a dose do campeão**; 0 é bypass EXATO (bit a bit) |
| preset de qualidade | 4 nomes | ver tabela abaixo |

### 5. Persistência (fica no app)

O worker é **sem estado por decisão**: o mesmo comando dá o mesmo
resultado sempre, e um lote não depende do que alguém clicou antes. A
última escolha do usuário (força e preset) é preferência de UI e mora no
Media Hub, como qualquer outra.

---

## Os presets de qualidade

Medidos em 19/08, clipe de 60 s, filtro + VSR 2× na 5080:

| `--quality` | qp | 60 s | episódio de 23,6 min |
|---|---|---|---|
| `lossless` | tune=lossless | 1,9 GB | 44,7 GB |
| `master` | 12 | 190 MB | 4,5 GB |
| `entrega` | 15 | 94 MB | 2,2 GB |
| `previa` | 25 | 18 MB | 0,4 GB |
| *(padrão, sem flag)* | 21 | 31 MB | 0,7 GB |

**O tempo é plano de qp 12 a qp 30** (35,6 a 36,6 s — dentro do ruído).
Só o `lossless` custa (+22%). Ou seja: o degrau é decisão de **tamanho**,
não de velocidade. Não existe motivo de performance para entregar
comprimido, e a UI pode dizer isso ao usuário.

Duas regras que o worker aplica sozinho:

- **Saída por cano ⇒ `lossless`.** Cano é intermediário por construção;
  alguém adiante vai encodar de novo. Não afeta o Media Hub, que escreve
  em arquivo.
- **Flag explícita ganha do preset.** `--quality master --nvenc-qp 20`
  dá 20.

---

## Limites declarados — leia antes de liberar

### O filtro é 8 bits

Entrada de **10 bits (P010) com `--cc-blob` = ERRO**, não bypass
silencioso. O worker sai com mensagem clara. Se o Media Hub aceita fonte
10 bits, a UI precisa ou desabilitar o filtro nesse caso ou converter
antes.

### THDR funciona, mas só desde 19/08

O ramo do THDR **ignorava o filtro em silêncio** até hoje — aceitava
`--cc-blob`, registrava "CodecClean ON" e não aplicava nada. Como o
Media Hub liga THDR quando o usuário pede HDR (`if !hdr { --no-thdr }`,
`rtx.rs:713`), esse era o caminho mais provável de o usuário cair.
Consertado e coberto por gate. **Não use binário mais velho que
`8255d70`.**

### Latência de 3 quadros

A janela é de 7 quadros centrada, então o quadro que sai é 3 atrás do
que entra. O worker resolve isso internamente: contagem e PTS saem
idênticos ao caminho sem filtro (provado por gate). **Não afeta o app** —
só saiba que os 3 primeiros `process()` não produzem saída, e a barra de
progresso já reflete isso.

### Resolução fora de 720p: NÃO MEDIDO

O modelo foi treinado em 720p e os mapas de condicionamento têm grade de
bloco fixa (4/8/16 px). O comportamento em 1080p, 480p ou 4K **não foi
medido**. Num app onde o usuário joga qualquer arquivo, isso é o maior
risco em aberto: o filtro não vai falhar, vai fazer *outra coisa*, e isso
não aparece como erro.

**Recomendação: medir antes de liberar**, e se a dose fugir, reamostrar
para 720p antes de filtrar ou avisar na UI.

---

## O que já está provado

| gate | o que cobra | resultado |
|---|---|---|
| `codecclean_convgate.cu` | kernel novo × ingênuo, **bit a bit**, 8 formas | 0 diferenças |
| `codecclean_gate.cu` (E1) | rede inteira × golden de Python | 0,00140 luma (barra 0,5) |
| `codecclean_wingate.cu` | 40 entram / 40 saem, ordem, bypass k=0 | exato |
| `eval/gate_worker_cc.py` | contagem, PTS, k=0 bit-idêntico — nos dois regimes de THDR | passa |
| `eval/gate_worker_luma.py` | luma do worker × do PyTorch | mediana 0,0000, viés 0,0089 |
| `eval/gate_presets.py` | cada preset resolve no que promete | 9/9 |

E o veredito do editor, quadro a quadro no DaVinci: o caminho em CUDA é
**indistinguível** do de PyTorch, com diferença apenas no dither (que é
ruído de média zero por construção — o viés de brilho medido é −0,000).

---

## Desempenho, para calibrar expectativa

Cadeia inteira (filtro + VSR 2× Q4 + deband + entrega), episódio de
23,6 min na 5080: **14,8 min**, contra 19,8 min do caminho antigo de dois
processos — e sem escrever intermediário nenhum em disco.

O filtro sozinho custa **13,7 ms/quadro** dentro do worker. Sem ele, o
worker faz 10,4 ms/quadro.
