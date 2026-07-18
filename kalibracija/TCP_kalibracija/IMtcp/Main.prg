Function main
    ' Deklaracija spremenljivk (usklajena s Pythonom)
    Real dx, dy, dz, rx, ry, rz
    Int32 type
    String line$
    String command$(7) ' Polje za razclenjevanje prejetih podatkov
    Integer i
    Integer abortFlag  ' Zastavica za varen izhod ob napaki kamere
    
    ' DODANO: Spremenljivke za TCP kalibracijo (Koda 3)
    Real cx, cy, cz, cu, cv, cw
    String msg$
    
    ' Zacetne nastavitve robota
    Motor On
    Power High
    Speed 50
    Accel 10, 10
    SpeedS 500
    AccelS 300, 300
    
    Go P0 +Z(30)
    
    Do
        Print "--- Cakam na povezavo s Pythonom na portu 5000... ---"
        CloseNet #201
        Wait 0.5
      
        OpenNet #201 As Client
        WaitNet #201
        
        Print "--- Povezava vzpostavljena! Zacenjam sekvenco... ---"
        
        Do
            If ChkNet(201) > 0 Then
                
                Input #201, line$
               
                ParseStr line$, command$(), " "
                type = Val(command$(0)) ' Prvi parameter je vedno KODA UKAZA
                
                ' --- KODA 2: Napaka / Premik izdelka (Abort) pred zacetkom ---
                If type = 2 Then
                    Print "[Robot] Kamera javlja PREMIK IZDELKA! Takoj prekinjam serijo!"
                    Exit Do ' Izstopimo iz notranje zanke
                EndIf
                
                ' --- KODA 3: Kalibracija TCP (4 tocke) ---
                If type = 3 Then
                    Print "=== ZACETEK TCP KALIBRACIJE (4 TOCKO VNOS) ==="
                    
                    For i = 1 To 4
                        Print "Zapogaj robota do konice pod kotom ", i, " in pritisni ENTER v Epson konzoli..."
                        
                        ' Program se ustavi in caka na pritisk tipke Enter
                        Input line$ 
                        
                        ' KLJUCNO: Preklopimo na Tool 0, saj želimo izmeriti tocno pozicijo flanše!
                        Tool 0
                        cx = CX(CurPos)
                        cy = CY(CurPos)
                        cz = CZ(CurPos)
                        cu = CU(CurPos)
                        cv = CV(CurPos)
                        cw = CW(CurPos)
                        
                        ' Sestavimo niz v formatu "x,y,z,u,v,w"
                        msg$ = Str$(cx) + "," + Str$(cy) + "," + Str$(cz) + "," + Str$(cu) + "," + Str$(cv) + "," + Str$(cw)
                        
                        ' Pošljemo tocko Pythonu preko socketa
                        Write #201, msg$
                        Print "[Robot] Tocka ", i, " poslana Pythonu: ", msg$
                    Next
                    
                    Print "=== TCP KALIBRACIJA NA ROBOTU ZAKLJUCENA ==="
                EndIf
                
                ' --- KODA 1: Prejet nov koordinatni sistem za zacetek cikla ---
                If type = 1 Then
                    
                    dx = Val(command$(2))
                    dy = Val(command$(3))
                    dz = Val(command$(4))
                    rx = Val(command$(5))
                    ry = Val(command$(6))
                    rz = Val(command$(7))
                    
                    Print "Prejel koordinatni sistem. Nastavljam Local 1..."
                    
                    ' Nastavimo nagnjeno mizo / kos
                    Local 1, XY(dx, dy, dz, rx, ry, rz)
                    
                    ' Dvignemo se nad zacetno tocko P0 glede na Local 1
                    Move P0 +Z(50) /1
                    Wait 1 ' Stabilizacija
                    
                    ' Ponastavimo zastavico pred zacetkom poti
                    abortFlag = 0
                    
                    ' ODREMO POT OD P0 DO P4
                    For i = 0 To 4
                        Print "Premik na tocko P", i
                        Move P(i) /1
                        
                        ' Vmes med premiki neprestano preverjamo, ce je Python poslal kodo za abort (2)
                        If ChkNet(201) > 0 Then
                            Input #201, line$
                            
                            ParseStr line$, command$(), " "
                            type = Val(command$(0))
                            
                            If type = 2 Then
                                Print "[Robot] Kamera javlja PREMIK IZDELKA! Prekinjam delo na P", i
                                abortFlag = 1 ' Nastavimo zastavico za abort
                                Exit For ' Izstopimo iz For zanke
                            EndIf
                        EndIf
                    Next
                    
                    ' Ce smo morali prekiniti delo, takoj zapustimo notranjo zanko
                    If abortFlag = 1 Then
                        Exit Do
                    EndIf
                    
                    Write #201, "1"
                    Print "[Robot] Nanos uspe?no zakljucen!"
                EndIf
                
            EndIf
            
            Wait 0.02 ' Kratka pavza za razbremenitev krmilnika (20 ms)
        Loop ' Konec notranje zanke (Loop za tocke)
        
        CloseNet #201
        Print "--- Povezava zaprta. Pripravljen na nov cikel. ---"
        Wait 1.0
        
    Loop

Fend
