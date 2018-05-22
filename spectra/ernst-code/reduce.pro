

pro reduce_image,name,flat,distmap,y0,y1,yc,w0,unit,app
   printf,unit,name
  
   ;read in data and correct for crosstalk between quadrants.
   img=crosstalk_corr(mrdfits('data/'+name,/sil))
   sky=img*0.
   bpm=fix(img*0)
   
   a2=img
   ;Mask the star
   mask=make_mask(a2,y0,y1,yc,w0)
   ;Calculate the initial sky-background
   sky2=skysub(a2,distmap,mask)
   ;Do an initial bad-pixel removal in the part away from the star
   rm_badpix,a2,sky2,mask,nbpix1,bpm0
   ;Refit the sky-background to the image without bad pixels.
   sky2=skysub(a2,distmap,mask)

   ;perform sky subtraction
   im=a2-sky2

   ;Trace the stellar centroid
   get_centroid,im,center
   ;Remove bad pixels/cosmic rays
   rm_cosmics,im,center,ncosmics,bpm0
   printf,unit,i,nbpix1,ncosmics,format='(" - ",i1,"  ",i5,"  ",i5)'
   ;Trace the stellar centroid after cosmic removal
   get_centroid,im,center

   ;Do aperture photometry, replace with optimal extraction if desired.
   flux=dblarr(1024,21)
   for i=0,1023 do begin
      for j=0,20 do begin
         flux[i,j]=total(im[i,round(j-app):round(j+app)])
      endfor
   endfor
   
   
   
   bpm=bpm0
   img=im
   sky=sky2


   writefits,'reduced/sc'+name,img
   writefits,'reduced/bpm_'+name,byte(bpm)
   writefits,'reduced/sky_'+name,sky
   writefits,'reduced/flux_'+name,flux

end






pro reduce
   get_lun,unit

   ;open file to write the number of bad pixels to
   openw,unit,'cm_stats.txt'
   ;read in a list of files to reduce
   readcol,'datalist',names,format='a'
   n_frm=n_elements(names)

   

   ; read in the fits file with the distortions of the detector
   ; In general offset in position with respect to row 512 is used.
   ; Map the distortion by tracing the arc lines in the spatial direction
   ; and fit a 2d surface to the offset of the arc lines with respect
   ; to the centre row.
   distmap=readfits('calib/distmap.fits')

   ;Read in flatfield
   flat=mrdfits('calib/flat.fits')
   
   y0=11   ;;Lower useable row
   y1=489  ;;Upper useable row 
   w0=30.  ;;Width for mask for stars
   yc=252. ;;Initial guess of the row where the star is on
   app=5+dindgen(21) ;;Aperture for aperture photometry
   t0=systime(/sec)
   for i=0,n_frm-1 do begin
      t1=systime(/sec)
      print,'starting on frame:',i,'  ',systime(/sec)-t0
      reduce_image,names[i],flat,distmap,y0,y1,yc,w0,unit,app
      
      print,systime(/sec)-t1,' sec'



   endfor 
   close,unit
end
