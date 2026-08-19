# Gera build_version.h a cada build. Roda como -P, fora do configure,
# que e o que impede a string de congelar.
#
# Ordem de preferencia da identidade:
#   1. git describe --tags --always --dirty   (identifica o CODIGO)
#   2. data/hora                              (se nao houver git)
# O sufixo de data entra sempre, para dois builds da mesma arvore ainda
# se distinguirem no log de um usuario.
set(VER "")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
    WORKING_DIRECTORY ${SRC}
    OUTPUT_VARIABLE VER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
string(TIMESTAMP STAMP "%Y.%m.%d.%H%M" UTC)
if(VER STREQUAL "")
  set(FULL "${STAMP}")
else()
  set(FULL "${VER} (${STAMP})")
endif()
set(TXT "#pragma once\n#define BUILD_VERSION \"${FULL}\"\n")
if(EXISTS ${OUT})
  file(READ ${OUT} OLD)
else()
  set(OLD "")
endif()
# So escreve se mudou: escrever sempre forcaria relink a cada build.
if(NOT OLD STREQUAL TXT)
  file(WRITE ${OUT} "${TXT}")
endif()
