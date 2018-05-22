


function skysub,img,shift,mask

;; Simplified after Kelson 2003  using IDLUTILS' BSPLINE_FIT

   ny=float(n_elements(img[0,*]))
   xx=fltarr(1024.*ny)
   x2=fltarr(1024.*ny)
   ff=fltarr(1024.*ny)
   mm=fltarr(1024.*ny)
   x00=findgen(1024.)

   xx[0:1023]=x00-shift[0,*] & for i=1,ny-1 do xx[1024l*i:1024l*(i+1)-1]=x00-shift[*,i]
   x2[0:1023]=0. & for i=1,ny-1 do x2[1024l*i:1024l*(i+1)-1]=i
   ff[0:1023]=img[*,0] & for i=1,ny-1 do ff[1024l*i:1024l*(i+1)-1]=img[*,i]
   mm[0:1023]=mask[*,0] & for i=1,ny-1 do mm[1024l*i:1024l*(i+1)-1]=mask[*,i] 
   s1=sort(xx)
   s1=s1[where(mm[s1])]
   m1=median(ff[s1],21)
   s=7.5*mad(ff[s1]-m1,med=med)
   s1=s1[where(abs(ff[s1]-m1)/s le 1.)]

   count=1d5
   while count gt 0 do begin
      errcode=bspline_fit(xx[s1],ff[s1],ff[s1]*0d0+1d0,sset,full=x00,nord=4,yfit=yfit,x2=x2[s1],npoly=2)
      if errcode ne 0 then stop
      mm=7.5*mad(ff[s1]-yfit,med=med)
      wb=where(abs(ff[s1]-yfit-med)/mm gt 1.,count,compl=wg)
      s1=s1[wg]
   endwhile

   yf=BSPLINE_VALU(xx,sset,x2=x2)
   sky0=img
   for i=0.,ny-1 do sky0[*,i]=yf[i*1024l:(i+1)*1024l-1.]
   if n_elements(where(sset.coeff)) le 100 then stop

   return,sky0
end



