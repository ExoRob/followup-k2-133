
pro rm_badpix,img0,sky,mask,nbpix,bpm
   bpm=fix(img0*0)
   img=img0
   ssub=img-sky
   M_img=median(ssub,9)
   d_img=ssub-m_img
   d_img=d_img*mask

   y=histogram(d_img[where(d_img ne 0.)],loc=x)
   r=gaussfit(x,y,aa,nterm=3)
   ww=where(d_img gt 9.*aa[2])
   nbpix=0
   if ww[0] ne -1 then begin
      img0[ww]=sky[ww]+m_img[ww]
      bpm[ww]=1
      nbpix=n_elements(ww)
   endif
end
