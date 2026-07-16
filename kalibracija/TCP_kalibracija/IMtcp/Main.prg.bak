Function main
    ' Deklaracija spremenljivk
    Real x, y, z, u, v, w
    String msg$
    String vnos$
    
    ' Preventiva: Zapremo port pred zagonom, da sprostimo vticnico
    CloseNet #201
    Wait 0.5
    
    Print "--- Robot (Client) se povezuje na Python strežnik (Port 201)... ---"
    
    ' Odpremo port 201 kot CLIENT in pocakamo na vzpostavitev povezave
    OpenNet #201 As Client
    WaitNet #201
    
    Print "--- Uspešno povezan na Python! Zacenjam zanko za zajem tock... ---"
    
   Do
        ' 1. Izpišemo navodilo za operaterja
        Print "Zapeljaj robota do konice in pritisni ENTER v konzoli (ali vpisi 'q' za izhod):"
        
        ' 2. Zaženemo Input, ki v konzoli izpiše "?" in pocaka na Enter
        Input vnos$
      
    
        ' Ce uporabnik vpiše 'q', pošljemo signal za izhod in zakljucimo program
        If LCase$(vnos$) = "q" Then
            Write #201, "exit"
            Exit Do
        EndIf
        Move P1 /1
        ' Preberemo trenutno pozicijo in jo pošljemo...
        Tool 0
        x = CX(CurPos)
        y = CY(CurPos)
        z = CZ(CurPos)
        u = CU(CurPos)
        v = CV(CurPos)
        w = CW(CurPos)
        
        msg$ = Str$(x) + "," + Str$(y) + "," + Str$(z) + "," + Str$(u) + "," + Str$(v) + "," + Str$(w)
        Write #201, msg$
        
        Print "Tocka uspesno poslana: ", msg$
        Wait 0.1
    Loop
    
    ' Varno zapremo povezavo
    CloseNet #201
    Print "--- Povezava zaprta. Konec programa. ---"
Fend
