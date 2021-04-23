set WORKING_DIR=%~dp0
set OUTPUT_DIR=%WORKING_DIR%\gen\parser
set LIB_DIR=%WORKING_DIR%\kotlin-spec\grammar\src\main\antlr

java -jar "%WORKING_DIR%\antlr-4.9.2-complete.jar" -o "%OUTPUT_DIR%" -Dlanguage=Python3 -no-listener -no-visitor -lib "%LIB_DIR%" "%LIB_DIR%\UnicodeClasses.g4"
java -jar "%WORKING_DIR%\antlr-4.9.2-complete.jar" -o "%OUTPUT_DIR%" -Dlanguage=Python3 -no-listener -no-visitor -lib "%LIB_DIR%" "%LIB_DIR%\KotlinLexer.g4"
java -jar "%WORKING_DIR%\antlr-4.9.2-complete.jar" -o "%OUTPUT_DIR%" -Dlanguage=Python3 -no-listener -no-visitor -lib "%LIB_DIR%" "%LIB_DIR%\KotlinParser.g4"