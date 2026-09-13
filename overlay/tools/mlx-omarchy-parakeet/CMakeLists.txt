# Pinned-reference downloader CLI. A python program: installed, never
# compiled; network-only, no device access.
install(
  PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/mlx_omarchy_parakeet.py
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  RENAME mlx-omarchy-parakeet)
