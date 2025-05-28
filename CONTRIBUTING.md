# 🤝 Guía de Contribución – Carmen1 AI

¡Gracias por tu interés en contribuir a Carmen1 AI! Este proyecto está diseñado para ser simple, modular y extensible.

## 🧱 ¿Cómo contribuir?

1. **Haz un fork** del repositorio.
2. **Clona tu fork** en tu máquina local.
3. **Crea una nueva rama** para tu cambio:
   ```bash
   git checkout -b mi-mejora
   ```
4. **Realiza tus cambios y pruébalos.**
5. **Haz un commit y push** de tu rama:
   ```bash
   git commit -m "Agrega nueva funcionalidad X"
   git push origin mi-mejora
   ```
6. **Abre un Pull Request** desde tu fork hacia la rama `main` del repositorio original.

---

## ✅ Buenas prácticas

- Sigue la estructura y estilo existente del código.
- Documenta tus funciones con comentarios claros.
- Si modificas la interfaz, asegúrate de que sea compatible con ambos modos (`Prompt` y `Docs-IA`).
- Añade capturas o explicaciones si tu PR modifica el comportamiento visible.

---

## 📦 Instalación del entorno

```bash
conda create -n ai-docs python=3.10 -y
conda activate ai-docs
pip install -r requirements.txt
./start.sh
```

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT.