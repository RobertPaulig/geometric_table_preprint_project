# Сборка PDF

Ниже — минимальные и воспроизводимые способы собрать PDF без `latexmk`.

## Linux / macOS
Рекомендуемый вариант (без `latexmk`):
```bash
./scripts/build_pdf.sh
```

## Windows (MiKTeX)
Есть два пути:
1) Установить Perl (например, Strawberry Perl) и использовать `latexmk`.
2) Собрать без `latexmk`:
```powershell
.\scripts\build_pdf.ps1
```

## Примечания
- Скрипты делают 2 прогона `lualatex` (достаточно, если нет библиографии).
- Если будет добавлена библиография (`.bib`), вставьте `biber` между прогонами.
