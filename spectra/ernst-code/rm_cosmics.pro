pro rm_cosmics,img0,trace,ncosmics,bpm0,iter=iter,bpm=bpm,reg=reg,nim=nim
   img=img0
   im3=img
   for i=0,1023 do im3[i,*]/=total(img[i,(trace[i]-25+findgen(51))])
   if n_elements(bpm) ne n_elements(img) then bpm=img*0
   if not keyword_set(iter) then iter=0
   
   if iter eq 0 then ncosmics=0

   mt=floor(median(trace))
   dx=trace-mt
   x00=mt-110+findgen(221)

   im4=im3
   for j=31,1023-31 do begin
      xx=x00-dx[0]
      ff2=reform(im3[0,x00])
      for i=j-31,j+31 do begin
         xi=x00-dx[i]
         xx=[xx,xi]
         ff2=[ff2,reform(im3[i,(x00)])]
      endfor 
      s1=sort(xx)
      xx=xx[s1]
      ff2=ff2[s1]
      ww=where(abs(ff2-median(ff2,7)) le 1.5*15.*mad(ff2-median(ff2,7)),compl=wb)

      errcode=bspline_fit(xx[ww],ff2[ww],ff2*0.+1./31.,sset,full=x00,nord=4,yf=yfit)
      im4[j,x00]=BSPLINE_VALU(x00-dx[j],sset)
      if j eq 31 then for i=0,31 do im4[i,x00]=BSPLINE_VALU(x00-dx[i],sset) 
      if j ge 1023-31 then for i=1023-31,1023 do im4[i,x00]=BSPLINE_VALU(x00-dx[i],sset) 
   endfor 


   im5=img
   for i=0,1023 do begin
      l=ladfit(im4[i,x00],img[i,x00])
      im5[i,x00]=poly(im4[i,x00],l)
   endfor 

   ii=img-im5
   smap=fltarr(n_elements(x00))
   for i=0,n_elements(x00)-1 do smap[i]=1.5*mad(ii[*,x00[i]])
   smap[where(smap/min(smap) gt 1.4)]*=2.
   for i=0,n_elements(x00)-1 do ii[*,x00[i]]=ii[*,x00[i]]/smap[i]

   ww=where( ii gt 7.)
   if ww[0] ne -1 then begin
      bpm[ww]=1
      bpm0[ww]=1
      ncosmics=ncosmics+n_elements(ww)   
   endif 
   iter=iter+1


   w2=where(bpm)
   if w2[0] ne -1 then img[w2]=im5[w2]

   if ww[0] ne -1 and iter lt 5 then rm_cosmics,img,trace,ncosmics,bpm0,iter=iter,/reg
   imga=img0
   img0=img
   nim=img
   if not keyword_set(reg) then print,iter,n_elements(w2),ncosmics,total(bpm0)
end
