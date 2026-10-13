# .venv removido em 2026-08-29

Removido para reduzir a pressão no VHDX: o C: do host estava com 28,3 GB livres de 953 GB.
Eram 8,3 GB, todos regeneráveis de forma determinística.

Para restaurar antes de rodar o benchmark:

```bash
cd /opt/amb && uv sync
```

`uv.lock` (1,2 MB) está no repositório e fixa as versões, então o ambiente volta idêntico.
