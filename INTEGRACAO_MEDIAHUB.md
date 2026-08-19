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

O worker mora em `<app_data>/bin/` e o app checa presença com
`worker_path()` (`rtx.rs:242`).

> **CORREÇÃO (19/08 19:10).** A primeira versão deste documento dizia que
> o worker é "um sidecar baixado sob demanda" e que atualizar seria só
> publicar um binário. **Não é verdade hoje.** Verificado: `tools.rs` tem
> exatamente dois `ToolSpec` — ffmpeg e deno. **Não existe downloader do
> worker.** O `rtx_worker_status` só faz `is_file()`, e a UI diz "baixe o
> enhancer primeiro" sem rota atrás do botão.
>
> Eu escrevi aquilo a partir do *comentário* no topo do `rtx.rs` ("driven
> as a lazy-downloaded sidecar") e do lugar onde o arquivo mora — sem
> verificar que o downloader existia. Comentário descreve intenção;
> código descreve o que acontece.

**Então a rota de instalação precisa ser construída**, e ela tem dois
requisitos que não dá para deixar para depois:

1. **Marcador de versão.** O `ensure()` do `tools.rs` (linha 113) decide
   por *existência de arquivo*: `if is_file() { return }`. Publicar
   binário novo na mesma URL **não alcança quem já tem um**. O binário
   instalado na máquina do Gui hoje é de **5 de julho** (22.383.104 B) —
   o que ignora o filtro em silêncio no caminho do THDR.
2. **O blob junto.** 310 KB, mesmo diretório.

Para comparar versão, o worker agora carimba a identidade **do git, a
cada build**: `--version` reporta `v0.2.0-13-g997fb8d-dirty
(2026.08.19.2159)`. Antes ele reportava o horário do último `cmake
configure`, que congelava em builds incrementais — um exe linkado às
17:52 dizia 13:07. Se o instalador for comparar versão, ele precisava
disso primeiro.

Fonte: **github.com/guilhermebarony-coder/rtx-video-worker** (privado até
o LICENSE ser revisado; MIT, com o copyright do autor original
preservado). Binário local:
`F:\CLAUDE\rtx-worker-fork\build-mod\RTXVideoProcessor.exe` (30,4 MB). Vai junto, como já vai hoje: `nvngx_vsr.dll` e
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
Consertado e coberto por gate.

> **Piso de versão: `888255f`** — *"o filtro era IGNORADO EM SILÊNCIO no
> caminho do THDR"*. Qualquer binário anterior a esse commit tem o
> defeito, **inclusive `8255d70`**, que este documento indicava por
> engano até 19/08 18:40. Se o hash não bater com o que você vê no repo,
> confira pelo ASSUNTO do commit: é ele que define o piso, não o número.
>
> Recomendado na prática: compile do `main`.

### O filtro não existe no caminho de CPU

`--cc-blob` com o processamento em CPU = **ERRO**, não bypass silencioso.
O filtro é CUDA e não roda ali.

**O gatilho não é só a flag `--cpu`**:
`use_cuda_path = (hw_device_ctx != nullptr) && !cpuOnly`. Decode por
hardware que não inicialize — codec exótico, driver, arquivo estranho —
derruba para CPU sozinho. Até 19/08 esse caminho aceitava `--cc-blob`,
registrava "CodecClean ON" e entregava o vídeo sem filtro nenhum.

Para o app: se o worker sair com essa mensagem, é sinal de que o decode
por hardware falhou para aquele arquivo — vale surfaçar isso ao usuário
em vez de tratar como erro genérico.

### Latência de 3 quadros

A janela é de 7 quadros centrada, então o quadro que sai é 3 atrás do
que entra. O worker resolve isso internamente: contagem e PTS saem
idênticos ao caminho sem filtro (provado por gate). **Não afeta o app** —
só saiba que os 3 primeiros `process()` não produzem saída, e a barra de
progresso já reflete isso.

### Resolução fora de 720p: MEDIDO, e transfere

Era o maior risco em aberto — o modelo treinou em 720p e os mapas têm
grade de bloco fixa (4/8/16 px). Medido em 19/08 (`eval/gate_resolucao.py`:
master limpo → escala → VP9 2-pass com bitrate escalado por pixel, para a
severidade por bloco ficar comparável):

| resolução | dose \|d\| | viés | p99 | vs 720p |
|---|---|---|---|---|
| 480p | 1,778 | −1,640 | 6,0 | 1,02× |
| 720p | 1,735 | −1,644 | 5,0 | 1,00× |
| 1080p | 1,684 | −1,644 | 4,0 | 0,97× |
| 2160p | 1,646 | −1,641 | 3,0 | 0,95× |

**A dose é plana e o viés de brilho bate em três casas nas quatro.** O
mecanismo: bloco de compressão tem tamanho fixo em *pixel codificado*
(16×16 macrobloco, 8×8 e 4×4 transformada) em qualquer resolução, então a
grade dos mapas casa igual. O p99 cair de 6 para 3 é coerente — em 4K o
mesmo bloco cobre menos desenho.

**Nenhuma ação necessária na UI.** Não precisa travar em 720p nem avisar.

*Ainda não medido:* material 720p **esticado** para 1080p antes de
filtrar — ali o artefato fica 1,5× maior que a grade espera. Caso
diferente de "arquivo nativo em 1080p", que é o testado acima.

---

## Preview de um quadro (novo, 19/08)

A agulha para num quadro e sai a imagem filtrada, sem renderizar o vídeo:
`eval/preview_frame.py` recorta N−3..N+3, roda o worker pelo **caminho de
produção** e puxa o do meio.

**~1,2 s por quadro — e 1,12 s disso é o worker iniciando** (CUDA + NGX).
Filtrar os 7 quadros custa ~0,1 s. Num app que mantenha o processo vivo
entre chamadas, o slider de força atualiza praticamente ao vivo.

A flag `--frame-offset N` diz ao worker qual é o índice absoluto do
primeiro quadro do recorte. Sem ela o dither usaria a semente do recorte
e o preview sairia ±1 de luma diferente da entrega. Com ela o preview é
**bit a bit** o pixel que o render vai produzir — provado em
`eval/gate_preview.py`, 4 quadros, zero diferenças.

> **O preview não valida o render.** Defeito de fluxo — quadro perdido,
> duplicado, fora de ordem — é invisível num quadro único, por
> construção. Ele julga a DOSE; quem valida o render é o gate de
> integração.

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
