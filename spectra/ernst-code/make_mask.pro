function make_mask,img,y0,y1,yc,w0

   mask=img*0.
   mask[*,y0:y1]=1.
   
   t0=fltarr(1024)
   yy=findgen(51)-25.
   for i=0,1023 do begin
      co=img[i,yc+yy]
      r=gaussfit(yy+yc,co,aa,nterm=4,estimates=[max(co),yc,4.,median(img[i,y0:yc-50])])
      t0[i]=aa[1]
   endfor 
   mm=median(t0,11)
   ww=where(abs(t0-yc) ge 21.)
   if ww[0] ne -1 then t0[ww]=mm[ww]

   l=poly_fit(dindgen(n_elements(img[*,0])),t0,2)
   t1=poly(dindgen(n_elements(img[*,0])),l)
   for i=0,1023 do mask[i,t1[i]-w0:t1[i]+w0]=0
   
   return,mask
end
