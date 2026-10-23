Option Explicit

' Run a scheduled command without creating a visible console window.
' Wait for the child so Task Scheduler tracks its lifetime and exit code.
Dim args, shell, cmdLine, value, index, exitCode
Set args = WScript.Arguments
If args.Count = 0 Then WScript.Quit 87

cmdLine = ""
For index = 0 To args.Count - 1
    value = CStr(args(index))
    If InStr(value, Chr(34)) <> 0 Then WScript.Quit 87
    If index > 0 Then cmdLine = cmdLine & " "
    cmdLine = cmdLine & Chr(34) & value & Chr(34)
Next

Set shell = CreateObject("WScript.Shell")
exitCode = shell.Run(cmdLine, 0, True)
WScript.Quit exitCode
