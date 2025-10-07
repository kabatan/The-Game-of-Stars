# check_import.py

print("Attempting to import player2...")
try:
    # player2.py のインポートを試みる
    import player2
    print("SUCCESS: 'player2.py' was imported successfully!")
    print("--- Checking the function ---")
    
    # player2 モジュールの中に 'move2' という名前の関数が存在するかチェック
    if hasattr(player2, 'move2'):
        print("SUCCESS: Function 'move2' was found inside the module.")
    else:
        print("ERROR: 'player2.py' was imported, but function 'move2' was NOT found!")

except Exception as e:
    # インポート中に何らかのエラーが発生した場合、その内容を詳しく表示
    print("\n!!! FAILED to import 'player2.py'. Here is the error: !!!")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")