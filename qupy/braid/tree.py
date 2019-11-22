#!/usr/bin/env python3

"""
General purpose (rooted) Tree structre.
"""

from qupy.smap import SMap


class Tree(object):
    def __init__(self, root, sites=None, leaves=None, parent=None):
        self.root = root
        self.sites = set([root]) if sites is None else set(sites)
        # Note: leaves != all sites without children
        self.leaves = [root] if leaves is None else leaves # bdy
        self.parent = {root : None} if parent is None else parent # map site->site or None if root

    def clone(self):
        tree = Tree(self.root, set(self.sites), list(self.leaves), dict(self.parent))
        return tree

    @classmethod
    def arrow(cls, tgt, src):
        tree = cls(tgt, [src, tgt], [src], {tgt:None, src:tgt})
        return tree

    def depth_first_search(self):
        parent = self.parent
        count = dict((site, 0) for site in self.sites)
        active = set(self.sites)
        for site in self.sites:
            site = parent[site]
            if site is not None:
                count[site] += 1
                if site in active:
                    active.remove(site)
        while active:
            _active = set()
            for site in active:
                assert count[site] == 0
                yield site # <------- yield
                site = parent[site]
                if site is None:
                    continue
                assert count[site]
                count[site] -= 1
                if count[site] == 0:
                    _active.add(site)
            active = _active

    def __str__(self):
        return "Tree(%s, %s, %s, %s)"%(self.root, list(self.sites), self.leaves, self.parent)

    def str(self, l):
        smap = SMap()

        for i in range(l):
          for j in range(l):
            smap[2*i+1, 2*j+1] = '.'

        for site in self.sites:
            #i, j, d = site
            i, j = site[0], site[1]
            key = 2*i+1, 2*j+1

            if smap[key] not in '. ':
                smap[key] = '?'
            elif site==self.root:
                smap[key] = '+'
            else:
                smap[key] = '*'

            if site==self.root:
                continue
            src, tgt = site, self.parent[site]
            m = abs(src[0]-tgt[0])+abs(src[1]-tgt[1])
            assert m>0
            if m==1:
                if src[0]==tgt[0]:
                    key = 2*src[0]+1, src[1]+tgt[1]+1
                    smap[key] = '-'
                elif src[1]==tgt[1]:
                    key = src[0]+tgt[0]+1, src[1]*2+1
                    smap[key] = '|'
                else:
                    assert 0
            else:
                # XXX 
                pass

        return str(smap)

    def __len__(self):
        return len(self.sites)

    def add(self, tgt, src):
        "add an edge to tgt from src"
        assert tgt in self.sites
        assert src not in self.sites
        self.parent[src] = tgt
        self.sites.add(src)
        self.leaves.append(src)

    def __contains__(self, site):
        return site in self.sites

    def get_path(self, src, dest=None, include_src=False):
        "get path (list of sites) to dest (default:root)"
        assert src in self.sites
        if dest is not None and dest!=self.root:
            return self._get_path(src, dest)
        parent = self.parent
        path = [src] if include_src else []
        while 1:
            tgt = parent[src]
            if tgt is None:
                break
            path.append(tgt)
            src = tgt
        return path

    def _get_path(self, src, dest):
#        print "_get_path", src, dest
        assert src!=dest
        path1 = self.get_path(src, include_src=True) # src ... root
        assert path1[0] == src
        path2 = self.get_path(dest, include_src=True) # dest ... root
        assert path2[0] == dest
        #if src==self.root:
        #    return list(reversed(path2)) # <--------- return <---
        while len(path1)>1 and path2 and path1[-2:]==path2[-2:]:
#            print path1, path2
            path1.pop(-1)
            path2.pop(-1)
        if path1[-1]==path2[-1]:
            path1.pop(-1)
#        print path1, path2
        path2 = list(reversed(path2))
        path = path1 + path2
        assert path[0] == src, path
        assert path[-1] == dest, path
        assert len(set(path)) == len(path), path
        return path

    def contains(self, other):
        return self.sites.issuperset(other.sites)

    def set_root(self, root):
        #print "set_root", root
        assert root in self.sites
        path = self.get_path(root)
        prev = root
        self.root = root
        parent = self.parent
        parent[root] = None
        for site in path:
            parent[site] = prev
            prev = site

    def get_radius(self):
        sites = self.sites
        parent = self.parent
        mark = dict((site, 0) for site in sites)
        for src in sites:
            tgt = parent[src]
            r = mark[src]
            while tgt is not None:
                r += 1
                if mark[tgt] >= r:
                    break
                mark[tgt] = r
                tgt = parent[tgt]
        r = max(mark.values())
        return r

    def grow(self, nbd, count=1):
        # nbd : maps site->list of site
        for i in range(count):
            leaves = []
            parent = self.parent
            sites = self.sites
            for tgt in self.leaves:
                for src in nbd[tgt]:
                    #if src not in self.leaves and src not in self.sites:
                    if src not in sites:
                        assert src not in parent, self
                        parent[src] = tgt
                        sites.add(src)
                        leaves.append(src)
            self.leaves[:] = leaves

    def grow_path(self, nbd, site):
        "grow until we find site then return a path to site"
        sites = self.sites
        leaves = self.leaves
        if site not in sites:
            while 1:
                self.grow(nbd)
                if site in self.leaves:
                    break
                assert self.leaves, "%s: site %r not found"%(self, site)
        return self.get_path(site)

    def intersects(self, other):
        return not self.sites.isdisjoint(other.sites)

    def join(self, other):
        assert self.intersects(other)
        leaves = []
        for site in other.leaves:
            if site not in self.sites or site in self.leaves:
                leaves.append(site)
        for site in self.leaves:
            if site not in other.sites:
                leaves.append(site)
        assert len(set(leaves))==len(leaves) # unique
        parent = dict(other.parent)
        parent.update(self.parent)
        if other.root not in self.sites:
            # join other.root to self.root
            sites = self.sites.intersection(other.sites)
            tgt = list(sites)[0]
            path = [tgt] + other.get_path(tgt)
            assert len(path)>=2
            i = len(path)-1
            assert path[i] == other.root
            while path[i] not in sites:
                assert i>0
                parent[path[i]] = path[i-1]
                i -= 1
        tree = Tree(self.root, self.sites.union(other.sites), leaves, parent)
        return tree

    def check(self):
        sites = set(self.parent.keys())
        assert sites==self.sites
        count = 0
        for src in sites:
            while 1:
                tgt = self.parent[src]
                assert (tgt is None) == (src==self.root)
                if tgt is None:
                    break
                src = tgt
                count += 1
                assert count < len(self.parent)**2 # no loops


def test():

    tree = Tree('a')

    tree.add('a', 'b')
    tree.add('a', 'c')
    tree.add('a', 'd')
    tree.add('a', 'e')
    tree.add('c', 'f')
    tree.add('c', 'g')
    tree.add('c', 'h')
    tree.add('g', 'i')

    items = list(tree.depth_first_search())
    for site, parent in list(tree.parent.items()):
        if parent is None:
            continue
        assert items.index(site) < items.index(parent)

    print("OK")


if __name__ == "__main__":

    test()

