#!/usr/bin/env python3

import numpy

from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2


class Toric3D(object):
    def __init__(self, l):

        stars = []
        for i in range(l):
          for j in range(l):
            for k in range(l):
              stars.append((2*i, 2*j, 2*k))

        self.l = l
        self.keys = keys = []
        self.keymap = keymap = {}
        for i, j, k in stars:
            self.add_site(i+1, j, k)
            self.add_site(i, j+1, k)
            self.add_site(i, j, k+1)
        #print keymap
        #print keys

        n = len(keys)
        #print "n =", n
        assert n == 3*(l**3)
        assert keys[keymap[2*l-1, 0, 0]] == (2*l-1, 0, 0)
        assert keys[keymap[-1, 0, 0]] == (2*l-1, 0, 0)

        m = len(stars)
        Hz = zeros2(m-1, n)

        for row, (i, j, k) in enumerate(stars):
            if row==len(stars)-1:
                break
            #if row==4:
              #print shortstr(Hz[row]), (i, j, k)
              #for d in (-1, 1):
                #print keys[keymap[i+d, j, k]],
                #print keys[keymap[i, j+d, k]],
                #print keys[keymap[i, j, k+d]],
              #print
            
            for d in (-1, 1):
                Hz[row, keymap[i+d, j, k]] = 1
                Hz[row, keymap[i, j+d, k]] = 1
                Hz[row, keymap[i, j, k+d]] = 1

        #print shortstr(Hz)

        Hx = zeros2(3*m, n)
        row = 0
        square = [(1, 0), (0, 1), (1, 2), (2, 1)]
        for i, j, k in stars:

            #print i, j, k

            if i+2<2*l or j+2<2*l:
              for d1, d2 in square:
                #print (i+d1, j+d2, k),
                #print keys[keymap[i+d1, j+d2, k]],
                Hx[row, keymap[i+d1, j+d2, k]] = 1
              #print
              #print shortstr(Hx[row])
              #for q in range(m-1):
                #if dot2(Hx[row], Hz[q].transpose()).sum():
                    #print shortstr(Hz[q]), "XXX"
              assert dot2(Hx[row], Hz.transpose()).sum()==0
              assert dot2(Hx, Hz.transpose()).sum()==0
              if not row or solve(Hx[:row].transpose(), Hx[row]) is None:
                row += 1
              else:
                Hx[row] = 0
                
            if i+2<2*l or k+2<2*l:
              for d1, d2 in square:
                Hx[row, keymap[i+d1, j, k+d2]] = 1
              assert dot2(Hx, Hz.transpose()).sum()==0
              if solve(Hx[:row].transpose(), Hx[row]) is None:
                row += 1
              else:
                Hx[row] = 0
                
            if i==0 and (j+2<2*l or k+2<2*l):
              for (d1, d2) in square:
                Hx[row, keymap[i, j+d1, k+d2]] = 1
              assert dot2(Hx, Hz.transpose()).sum()==0
              if solve(Hx[:row].transpose(), Hx[row]) is None:
                row += 1
              else:
                Hx[row] = 0

        Hx = Hx[:row].copy()

        self.Hx = Hx
        self.Hz = Hz

    def add_site(self, i, j, k):
        l = self.l
        m = len(self.keys)
        i %= 2*l
        j %= 2*l
        k %= 2*l
        for di in (-2*l, 0, 2*l):
          for dj in (-2*l, 0, 2*l):
            for dk in (-2*l, 0, 2*l):
                self.keymap[i+di, j+dj, k+dk] = m
        self.keys.append((i, j, k))

    def strop(self, op):
        return shortstr(op)
          


class Toric2D(object):
    def __init__(self, l, allgen=False):
    
        keys = []
        keymap = {}
        for i in range(l):
          for j in range(l):
            for k in (0, 1):
                m = len(keys)
                keys.append((i, j, k))
                for di in (-l, 0, l):
                  for dj in (-l, 0, l):
                    keymap[i+di, j+dj, k] = m
    
        if l>2:
            assert keys[keymap[2, 1, 0]] == (2, 1, 0)
    
        if allgen:
            m = l**2 # rows (constraints)
        else:
            m = l**2-1 # rows (constraints)
        n = len(keys) # cols (bits)
        assert n == 2*(l**2)
    
        Lx = zeros2(2, n)
        Lz = zeros2(2, n)
        Hx = zeros2(m, n)
        Tz = zeros2(m, n)
        Hz = zeros2(m, n)
        Tx = zeros2(m, n)

        for i in range(l):
            Lx[0, keymap[i, l-1, 1]] = 1
            Lx[1, keymap[l-1, i, 0]] = 1
            Lz[0, keymap[0, i, 1]] = 1
            Lz[1, keymap[i, 0, 0]] = 1
    
        row = 0
        xmap = {}
        for i in range(l):
          for j in range(l):
            if (i, j)==(0, 0) and not allgen:
                continue
            Hx[row, keymap[i, j, 0]] = 1
            Hx[row, keymap[i, j, 1]] = 1
            Hx[row, keymap[i-1, j, 0]] = 1
            Hx[row, keymap[i, j-1, 1]] = 1
            xmap[i, j] = row
            i1 = i
            while i1>0:
                Tz[row, keymap[i1-1, j, 0]] = 1
                i1 -= 1
            j1 = j
            while j1>0:
                Tz[row, keymap[i1, j1-1, 1]] = 1
                j1 -= 1
            row += 1

        row = 0
        zmap = {}
        for i in range(l):
          for j in range(l):
            if i==l-1 and j==l-1 and not allgen:
                continue
            Hz[row, keymap[i, j, 0]] = 1
            Hz[row, keymap[i, j, 1]] = 1
            Hz[row, keymap[i+1, j, 1]] = 1
            Hz[row, keymap[i, j+1, 0]] = 1
            zmap[i, j] = row
            i1 = i
            while i1<l-1:
                Tx[row, keymap[i1+1, j, 1]] = 1
                i1 += 1
            j1 = j
            while j1<l-1:
                Tx[row, keymap[i1, j1+1, 0]] = 1
                j1 += 1
            row += 1

        self.mx = self.mz = m
        self.n = n
        self.Lx = Lx
        self.Lz = Lz
        self.Hx = Hx
        self.Tz = Tz
        self.Hz = Hz
        self.Tx = Tx
        self.keys = keys
        self.keymap = keymap
        self.xmap = xmap
        self.zmap = zmap
        self.l = l

    def get_code(self, **kw):
        from qupy.ldpc.css import CSSCode
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        code = CSSCode(
            Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, 
            Hz=Hz, Tx=Tx, **kw)
        code.__dict__.update(self.__dict__)
        return code 

    def translate(self, di, dj):
        mx, mz = self.mx, self.mz
        n = self.n
        nmap = self.keymap
        xmap = self.xmap
        zmap = self.zmap
        l = self.l
        Dx = zeros2(mx, mx)
        D = zeros2(n, n)
        Dz = zeros2(mz, mz)
        #print nmap
        for i in range(l):
          for j in range(l):
            Dx[xmap[(i+di)%l, (j+dj)%l], xmap[i, j]] = 1
            Dz[zmap[(i+di)%l, (j+dj)%l], zmap[i, j]] = 1
        for i, j, k in self.keys:
            D[nmap[(i+di)%l, (j+dj)%l, k], nmap[i, j, k]] = 1
        return Dz, D, Dx

    def strop(self, u, fancy=False):
        m = SMap()
        n = u.shape[0]
        for i in range(n):
            c = str(min(1, u[i]))
            i, j, k = self.keys[i]
            row = 2*i + (1-k)
            col = 4*j + 2*k
            #if i%2==0 and j%2==0 and k==0:
            if fancy:
                if k==0:
                    m[row-1, col] = '+'
                if c=='I':
                    c = '|-'[k]
            m[row, col] = c
        return str(m).replace('0', '.')


class Surface(object):
    def __init__(self, l, allgen=False):
    
        keys = []
        keymap = {}
        for i in range(l):
          for j in range(l):
            ks = (0, 1) if i+1<l and j+1<l else (0,)
            for k in ks:
                m = len(keys)
                keys.append((i, j, k))
                keymap[i, j, k] = m
    
        if l>2:
            assert keys[keymap[2, 1, 0]] == (2, 1, 0)
    
        m = l*(l-1) # rows (constraints)

        n = len(keys) # cols (bits)
        assert n == 2*((l-1)**2) + 2*(l-1) + 1

        assert 2*m+1==n
    
        self.mx = self.mz = m
        self.n = n
        self.keys = keys
        self.keymap = keymap

        #Lx = zeros2(2, n)
        #Lz = zeros2(2, n)
        Hx = zeros2(m, n)
        #Tz = zeros2(m, n)
        Hz = zeros2(m, n)
        #Tx = zeros2(m, n)

        #for i in range(l):
        #    Lx[0, keymap[i, l-1, 1]] = 1
        #    Lx[1, keymap[l-1, i, 0]] = 1
        #    Lz[0, keymap[0, i, 1]] = 1
        #    Lz[1, keymap[i, 0, 0]] = 1
    
        row = 0
        xmap = {}
        for i in range(l):
          for j in range(l-1):
            Hx[row, keymap[i, j, 0]] = 1
            if i+1<l:
                Hx[row, keymap[i, j, 1]] = 1
            if i:
                Hx[row, keymap[i-1, j, 1]] = 1
            if j+1<l:
                Hx[row, keymap[i, j+1, 0]] = 1
            xmap[i, j] = row
        #    i1 = i
        #    while i1>0:
        #        Tz[row, keymap[i1-1, j, 0]] = 1
        #        i1 -= 1
        #    j1 = j
        #    while j1>0:
        #        Tz[row, keymap[i1, j1-1, 1]] = 1
        #        j1 -= 1
            #print self.strop(Hx[row])
            #print
            row += 1
        assert row==m, (row, m)
        #print shortstr(Hx)

        row = 0
        zmap = {}
        for i in range(l-1):
          for j in range(l):
            Hz[row, keymap[i, j, 0]] = 1
            if j+1<l:
                Hz[row, keymap[i, j, 1]] = 1
            if i+1<l:
                Hz[row, keymap[i+1, j, 0]] = 1
            if j:
                Hz[row, keymap[i, j-1, 1]] = 1
            zmap[i, j] = row
        #    i1 = i
        #    while i1<l-1:
        #        Tx[row, keymap[i1+1, j, 1]] = 1
        #        i1 += 1
        #    j1 = j
        #    while j1<l-1:
        #        Tx[row, keymap[i1, j1+1, 0]] = 1
        #        j1 += 1
            #print self.strop(Hz[row])
            #print
            row += 1
        assert row==m

        #self.Lx = Lx
        #self.Lz = Lz
        self.Hx = Hx
        #self.Tz = Tz
        self.Hz = Hz
        #self.Tx = Tx
        self.xmap = xmap
        self.zmap = zmap
        self.l = l

    def get_code(self, **kw):
        from qupy.ldpc.css import CSSCode
        #Lx, Lz, Hx, Tz, Hz, Tx = (
        #    self.Lx, self.Lz, self.Hx,
        #    self.Tz, self.Hz, self.Tx)
        code = CSSCode(Hx=self.Hx, Hz=self.Hz, **kw)
        code.__dict__.update(self.__dict__)
        return code 

    def strop(self, u, fancy=False):
        m = SMap()
        n = u.shape[0]
        for i in range(n):
            c = str(min(1, u[i]))
            i, j, k = self.keys[i]
            row = 2*i + k
            col = 4*j + 2*k
            #if i%2==0 and j%2==0 and k==0:
            if fancy:
                if k==0:
                    m[row-1, col] = '+'
                if c=='I':
                    c = '|-'[k]
            m[row, col] = c
        return str(m).replace('0', '.')


