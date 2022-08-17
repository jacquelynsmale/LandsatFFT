function Ix = arImgFFT(validDomain, Ix)

  % remove across-track stripes in image

  % find the 4 corners of the image

  [m, n] = size(validDomain);
  s = regionprops(validDomain,'centroid');

  % sometimes there is more than one region (isloated
  % valid pixels) so pick largest region
  if length(s) > 1
      a = regionprops(validDomain,'area');
      idx = max([a.Area]) == [a.Area];
      s = s(idx);
      clear a idx
  end
  foo = false([m,n]);
  foo(round(s.Centroid(2)),round(s.Centroid(1))) = true;
  foo = bwdist(foo);
  foo(~validDomain) = 0;

  mC = round(m/2);
  nC = round(n/2);

  % bottom left
  [blR, blC] = find((max(max(foo(1:mC,1:nC))) == foo(1:mC,1:nC)),1, 'first');

  % bottom right
  [brR, brC] = find((max(max(foo(1:mC,(nC+1):end))) == foo(1:mC,(nC+1):end)),1, 'first');
  brC = brC+nC;

  % top left
  [tlR, tlC] = find((max(max(foo((mC+1):end,1:nC))) == foo((mC+1):end,1:nC)),1, 'first');
  tlR = tlR + mC;

  % top right
  [trR, trC] = find((max(max(foo((mC+1):end,(nC+1):end))) == foo((mC+1):end,(nC+1):end)),1, 'first');
  trR = trR + mC;
  trC = trC + nC;

  % figure
  % imagesc(validDomain)
  % hold on
  % plot(blC,blR, 'r*');
  % plot(brC,brR, 'g*');
  % plot(tlC, tlR, 'b*');
  % plot(trC,trR,  'k*');
  % plot(s.Centroid(1),s.Centroid(2),  'c*');

  % deteremine distance between corner points
  corners = [blC, blR; brC, brR; tlC, tlR; trC, trR];
  sep = pdist(corners,'euclidean');

  if any(sep < mC)
      % corners are likely in middle... try diferent
      % approach

      % bottom left
      [blR, blC] = find((max(max(foo(1:(round(mC/2)),1:end))) == foo(1:(round(mC/2)),1:end)), 1, 'first');

      % bottom right
      [brR, brC] = find((max(max(foo(1:end,round(nC/2*3):end))) == foo(1:end,round(nC/2*3):end)), 1, 'first');
      brC = brC+round(nC/2*3)-1;

      % top left
      [tlR, tlC] = find((max(max(foo(round(mC/2*3):end,1:end))) == foo(round(mC/2*3):end,1:end)),1, 'first');
      tlR = tlR + round(mC/2*3) -1;

      % top right
      [trR, trC] = find((max(max(foo(1:end,1:round(nC/2)))) == foo(1:end,1:round(nC/2))), 1, 'first');

      corners = [blC, blR; brC, brR; tlC, tlR; trC, trR];
      sep = pdist(corners,'euclidean');
      if any(sep < mC)
          % at least 2 points are too close to each other
          error('two or more recovered image corner locations are too close to eachother\n add to cleanLandsatDataDir corruptImages list then run cleanLandsatDataDir:\n %s'...
              , fooFileName)
      end

  end

  clear foo

  slope1 = atan((brR - blR) / (brC - blC));
  slope2 = atan((trR - tlR) / (trC - tlC));

  slope3 = atan((tlR - blR) / (blC - tlC));
  slope4 = atan((brR - trR) / (trC - brC));

  fooA = false(size(validDomain));
  center = round(size(fooA)/2);

  fooA(floor(center(1)) - 70 : floor(center(1)) + 70, :) = 1; % create mask - horizontal line
  fooA(:,(floor(center(2)) - 100) : (floor(center(2)) + 100)) = 0; % with zeros at the center

  % slope will be reduced is image is truncated at corners
  % so choose max slope
  A = imrotate(fooA,rad2deg(max([slope1,slope2])), 'crop');
  B = imrotate(fooA,rad2deg(max([slope3,slope4])), 'crop');

  % shift mask to center
  shiftCtr = round([s.Centroid(2), s.Centroid(1)]) - center ;
  tform = affine2d([1 0 0; 0 1 0; shiftCtr(2) shiftCtr(1) 1]);
  [A,~] = imwarp(A,tform);
  [B,~] = imwarp(B,tform);

  %Perform 2D FFTs
  for k = 1:length(size(Ix,3))

      fftIm = Ix(:,:,k);
      fftIm(fftIm>3) = 3;
      fftIm(fftIm<-3) = -3;
      fftIm(isnan(fftIm)) = 0;
      fftIm = fftshift(fft2(double(fftIm)));
      P = abs(fftIm);
      mP =  mean(P(:));
      stdP = std(P(:));
      P = (P-mP) > 10*stdP;
      sA = sum(P(A));
      sB = sum(P(B));

      % imagesc(P+A)

      if (sA/sB >= 2 || sB/sA >= 2) && (sA > 500 || sB > 500)
          if sA > sB
              mask = A;
          elseif sB > sA
              mask = B;
          end

          %figure; imshow(real(ifft2(ifftshift(fftIm .* (1-(mask ))))));
          foo1 = isnan(Ix(:,:,k));
          foo2 = real(ifft2(ifftshift(fftIm .* (1-(mask)))));
          foo2(foo1) = nan;

          %figure; imshow(foo2);caxis([-1.5 1.5])

          Ix(:,:,k) = foo2;

          clear fftIm mask foo1 foo2
      end
  end
