#!/usr/bin/env python3
"""
Скрипт для подготовки agent_system перед загрузкой в DataSphere.
Удаляет проблемную директорию с длинными путями используя robocopy (надежнее для Windows).
"""
import os
import subprocess
import sys
from pathlib import Path

def remove_with_robocopy(path):
    """Удаляет директорию используя robocopy (работает с длинными путями в Windows)"""
    if not path.exists():
        return True
    
    # Создаем пустую временную директорию
    temp_dir = path.parent / f"__temp_delete_{path.name}"
    try:
        temp_dir.mkdir(exist_ok=True)
    except:
        pass
    
    # Используем robocopy для удаления (копируем пустую директорию поверх целевой)
    try:
        result = subprocess.run(
            ["robocopy", str(temp_dir), str(path), "/MIR", "/NFL", "/NDL", "/NJH", "/NJS"],
            capture_output=True,
            text=True
        )
        # robocopy возвращает коды 0-7 для успешных операций, 8+ для ошибок
        if result.returncode <= 7:
            # Удаляем временную директорию
            try:
                temp_dir.rmdir()
            except:
                pass
            # Удаляем целевую директорию (теперь она пустая)
            try:
                path.rmdir()
            except:
                pass
            return True
    except Exception as e:
        print(f"  WARNING: robocopy не сработал: {e}")
    
    # Fallback: пробуем обычное удаление
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
        return True
    except:
        pass
    
    return False

def main():
    script_dir = Path(__file__).parent
    agent_system_dir = script_dir / "agent_system"
    problematic_dir = agent_system_dir / "environments" / "env_package" / "craftext" / "CrafText-super_igor_env_build"
    backup_dir = agent_system_dir / "agent_system_backup"
    
    if not agent_system_dir.exists():
        print(f"ERROR: {agent_system_dir} не найден")
        return 1
    
    # Удаляем backup если он есть внутри agent_system (чтобы не загружался в datasphere)
    if backup_dir.exists():
        print(f"Удаление backup директории из agent_system...")
        if remove_with_robocopy(backup_dir):
            print("✓ Backup директория удалена")
        else:
            print("⚠ Не удалось полностью удалить backup, но продолжаем...")
    
    # Удаляем проблемную директорию с длинными путями
    if problematic_dir.exists():
        print(f"Удаление проблемной директории с длинными путями...")
        if remove_with_robocopy(problematic_dir):
            print("✓ Проблемная директория успешно удалена")
        else:
            print("ERROR: Не удалось удалить проблемную директорию")
            return 1
    else:
        print("Проблемная директория не найдена, пропускаем")
    
    print("Подготовка завершена!")
    return 0

if __name__ == "__main__":
    exit(main())
