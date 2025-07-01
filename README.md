# Instalación

Para instalar se recomienda crear un entorno virtual, activarlo y luego instalar las librerias.

```python
python3 venv -m venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Descargar los pesos de `SAM 2.1` usando:
```bash
make checkpoint
```

o tambien descargando los pesos de este link:
```
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

>Nota: Cabe destacar que los pesos deben estar dentro de la carpeta `checkpoint` en la raiz del repo.

# Uso

Ejecutar las celdas del jupyter notebook