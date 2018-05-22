


pro get_centroid,img,center

   ny=n_elements(img[0,*])
   y=dindgen(ny)
   y=y[10:ny-1-1]
   x0=findgen(1024)
   xc=7.5+findgen(64)*16.

   parinfo=create_struct('fixed',0,'limited',[0,0],'limits',[0.,0.],'value',0.)
   parinfo=replicate(parinfo,6)

   ;;Do a zeroth order fit for the centroid and moffat parameters
   flux=total(img[10:1013,y],1)
   res=MPFITPEAK(y, flux, aa, NTERMS=6,/moffat)
   estimates=[max(flux),aa[1],aa[2],aa[3],aa[4],aa[5]]
   cc0=aa[1]
   parinfo[3].value=estimates[3]
   parinfo[3].fixed=0
   parinfo[3].limited=[1,1]
   parinfo[3].limits=estimates[3]+[-0.25,0.25]
   
   c=fltarr(64)
   for i=0,63 do begin
      flux=total(img[i*16:(i+1)*16-1,y],1)
      estimates=[max(flux),aa[1],aa[2],parinfo[3].value,aa[4],aa[5]]
      aa=0
      res=MPFITPEAK(y, flux, aa, NTERMS=6,/moffat,parinfo=parinfo,estimates=estimates)
      c[i]=aa[1]
   endfor 

   center=poly(x0,poly_fit(xc,c,4))



end 
