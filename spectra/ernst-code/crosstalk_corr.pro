function crosstalk_corr,img
   img2=img
   for j=0,511 do begin
      ct=double(total(img[*,j])+total(img[*,j+512]))
      img2[*,j]=img2[*,j]-1.4d-5*ct
      img2[*,j+512]=img2[*,j+512]-1.4d-5*ct
   endfor
   return,img2
end
