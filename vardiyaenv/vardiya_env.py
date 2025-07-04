# myenv/vardiyaenv/vardiya_env.py

import gym
from gym import spaces
import numpy as np
from typing import Optional
from gymnasium.envs.toy_text.utils import categorical_sample
from os import path
import time
from gymnasium.error import DependencyNotInstalled



#import gymnasium as gym
#from gymnasium import Env, spaces, utils



MAP = [
    "+---------------+",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "|               |",
    "+---------------+",
]
WINDOW_SIZE = (850, 850)


class VardiyaEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
    
    def __init__(self, render_mode: Optional[str] = None):
        #super(VardiyaEnv, self).__init__()
        self.desc = np.asarray(MAP, dtype="c")
        self.ii = 0
        
        self.aracsayisi = aracsayisi = 5
        self.personelsayisi = personelsayisi = 15
        
        rows = aracsayisi*3
        cols = personelsayisi
        self.renkmavi = renkmavi = (0,0,255)
        self.renkkirmizi = renkkirmizi = (255,0,0)
        self.locs =locs = [(i, j) for i in range(rows) for j in range(cols)]
        self.yerilistesi=[(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),
                          (renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),
                          (renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1)]
        self.stepsayac = -1
        
        #self.locs = locs = [(0, 0), (0, 1), (0, 2),(0,3),(0,4),
        #                    (1, 0), (1, 1), (1, 2),(1,3),(1,4),
        #                    (2, 0), (2, 1), (2, 2),(2,3),(2,4)]
        #self.locs_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        self.pers_color = (0, 0, 255)
        self.gezinti_color = (255,255,255)

        #              1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        ehliyetler = ['X','E','X','B','B','B','C','E','C','D','D','D','E','C','X']
        personelid = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.araclar = araclar = ['A','B','C','D','E','A','B','C','D','E','A','B','C','D','E']
        self.aracpersonel = aracpersonel = [0,0,0,0,0,
                                            0,0,0,0,0,
                                            0,0,0,0,0]
        #Sonradan eklendi
        self.aracpernerden = aracpernerden = [0,0,0,0,0,
                                              0,0,0,0,0,
                                              0,0,0,0,0]
        self.aracpersoneloncekigun = aracpersoneloncekigun = [1,4,7,10,2,
                                                              3,5,9,11,8,
                                                              3,6,9,12,13]

        self.aracpersoneldunsonvardiya = aracpersoneldunsonvardiya = aracpersoneloncekigun[-5:]
        
        terminated = False
       
        num_states = aracsayisi*3*personelsayisi
        num_actions = 3    #SağaGit  #İşaretle  #BirÖncekiVardiya
        num_rows = 3 #5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }

        for a in range(0,aracsayisi*3):  # A1 B1 C1 D1 E1  A2 B2 C2 D2 E2  A3 B3 C3 D3 E3
            for prs in range(0,personelsayisi):
                state = self.encode(a, prs)
                araccinsi = araclar[a] #A
                if state==0 : self.initial_state_distrib[0] += 1
                for action in range(0,num_actions):
                    
                    if (action==0): #Sağa Git
                        ya = a
                        yp = prs+1
                        reward = -2
                        terminated = False
                        if (yp%personelsayisi==0): #satır sonuna gelmişse
                            # İşaretleme var ise bir alt satıra 
                            # İşaretleme yok ise aynı state kal
                            if (aracpersonel[a]!=0): #işaretleme var
                                ya=a+1
                                yp=0
                                reward = -2
                                terminated = False
                                if ((ya%(aracsayisi*3))==0): #tüm satırlar bitmiş
                                    #Oyun tamamlandı
                                    ya=0
                                    yp=0
                                    reward = 50#500#50
                                    terminated = True
                            else: #işaretleme yok olduğu yerde kal
                                ya=a
                                yp=prs
                                reward = -3
                                terminated = False
                        
                        yenistate = self.encode(ya,yp)
                        self.P[state][action].append((1.0, yenistate, reward, terminated))
                        
                    elif (action==1):  #işaretle
                        if (personelid[prs] != 0 and aracpersonel[a]==0):  #personel var araca şoför atanmamış
                            if (ehliyetler[prs]==araccinsi): #Ehliyet uygun
                                #bir sonraki arac yani bir alt satır
                                aracpersonel[a] = prs+1  #ATAMA
                                personelid[prs] = 0 #bu personel bir araca atandı
                                ya=a+1
                                yp=0
                                reward = -1
                                terminated = False
                                if ((ya%(aracsayisi*3))==0): #Son satırda işaretleme yapılmış
                                    #Oyun tamamlandı
                                    ya=0
                                    yp=0
                                    reward = 50#500#50
                                    terminated = True
                            else:  # Ehliyet uygun değil aynı state kal
                                ya=a
                                yp=prs
                                reward = -3
                                terminated = False
                        else:  #mevcut satırdaki araç için personel önceden atanmış
                            ya=a
                            yp=prs
                            reward = -5
                            terminated = False
                            
                        yenistate = self.encode(ya,yp)
                        self.P[state][action].append((1.0, yenistate, reward, terminated)) 
                                
                    elif (action==2): #Bir önceki vardiyanın personelini işaretle
                        if (aracpersonel[a]!=0):  #mevcut satırdaki araç için personel önceden atanmış
                            ya=a
                            yp=prs
                            reward = -5
                            terminated = False
                        else: # o satırdaki araca atama yapılmamışsa
                            #araca uygun personeller tükendi ise bu seçenek değerlendirilmeli
                            #önce listedeki personel sonra önceki vardiye
                    
                            aracauygunpers = [iarac+1 for iarac, ehliyet in enumerate(ehliyetler) if ehliyet == araccinsi]
                            #aracauygunpers örneğin [1,3]
                            eslesmedurumu = any(i in personelid for i in aracauygunpers) #True-False
                            if eslesmedurumu==True: #daha atanmamış boşta personel var
                                ya=a
                                yp=prs
                                reward = -5
                                terminated = False
                            else: #boşta personel yok
                                #Bir önceki günün vardiyaları aracpersonelonceki dizisinde yer alıyor
                        
                                oncekivdy = a-aracsayisi
                                oncekivdypersoneli = 0
                                if (oncekivdy<0): #yeni günün ilk vardiyası demektir.
                                    oncekivdy = oncekivdy+5
                                    oncekivdypersoneli = aracpersoneldunsonvardiya[oncekivdy]
                                else:
                                    oncekivdypersoneli = aracpersonel[oncekivdy]
                                    
                                if (oncekivdypersoneli!=0): #önceki vardiyada personelid var.
                                    aracpersonel[a] = oncekivdypersoneli  #ATAMA
                                    aracpernerden[a] = 1 #önceki vardiyadan olanlar 1 normal olanlar 0
                                    #bir alt satıra geç
                                    ya = a+1
                                    yp=0
                                    reward = -1
                                    terminated = False
                                    if ((ya%(aracsayisi*3))==0): #Son satırda yapılmış
                                        #Oyun tamamlandı
                                        ya=0
                                        yp=0
                                        reward = 50#500
                                        terminated = True
                                else: #önceki vardiya için kimse atanmamış aynı yerde kal
                                    ya=a
                                    yp=prs
                                    reward = -5
                                    terminated = False
                                                  
                        yenistate = self.encode(ya,yp)
                        self.P[state][action].append((1.0, yenistate, reward, terminated))
                            
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None
        self.background_img2 = None


    def encode(self, a, b):
        sonuc = (a*self.personelsayisi)+b
        return sonuc

    def decode(self, i):
        out = []
        out.append(i % self.personelsayisi)
        i = i // self.personelsayisi
        out.append(i)
        assert 0 <= i < self.aracsayisi*3
        return reversed(out)

    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(2, dtype=np.int8)
        return mask


    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        '''
        # Bu if sonradan eklendi
        if self.render_mode == "ansi":
            self._render_text()
        '''
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p, "action_mask": self.action_mask(s)}        



    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.taxi_orientation = 0

        #self.yerilistesi = []
        #self.yerilistesi=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        self.renkmavi = renkmavi = (0,0,255)
        self.renkkirmizi = renkkirmizi = (255,0,0)
        self.yerilistesi=[(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),
                          (renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),
                          (renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1),(renkmavi,-1)]
        
        
        self.stepsayac = -1
        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)
            
    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            elif mode == "rgb_array":
                self.window = pygame.Surface(WINDOW_SIZE)

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.background_img2 is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.background_img2 = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        font = pygame.font.SysFont(None, 24)  # Yazı tipi ve boyutu

        

        iy=1
        for harf in (self.araclar):
            cell = (0 * self.cell_size[0], iy * self.cell_size[1])
            # 1. Hücre yüzeyi oluştur
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill((255,255,255))
    
            # 2. Ekrandaki konumunu al
            loc = self.get_surf_loc(cell)

            # 3. Harfi oluştur (örnek: "A")
            text_surface = font.render(str(harf), True, (0, 0, 0))  # Siyah renk
    
            # 4. Harfi hücreye ortalayarak ekle
            text_rect = text_surface.get_rect(center=(self.cell_size[0] // 2, self.cell_size[1] // 2))
            color_cell.blit(text_surface, text_rect)

            # 5. Son olarak hücreyi pencereye çiz
            self.window.blit(color_cell, cell)
            iy=iy+1

        #ix=1
        for ix in range(1,self.personelsayisi+1):
            cell = (ix * self.cell_size[0], 0 * self.cell_size[1])
            # 1. Hücre yüzeyi oluştur
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill((255,255,255))
    
            # 2. Ekrandaki konumunu al
            loc = self.get_surf_loc(cell)

            # 3. Harfi oluştur (örnek: "A")
            text_surface = font.render(str(ix), True, (0, 0, 0))  # Siyah renk
    
            # 4. Harfi hücreye ortalayarak ekle
            text_rect = text_surface.get_rect(center=(self.cell_size[0] // 2, self.cell_size[1] // 2))
            color_cell.blit(text_surface, text_rect)

            # 5. Son olarak hücreyi pencereye çiz
            self.window.blit(color_cell, cell)

        #print(self.yerilistesi)
        
        yerX,yerY = self.decode(self.s)
        yeri = self.encode(yerX,self.aracpersonel[yerX]-1)
        self.stepsayac= yerX-1
        if self.s == yeri:
            #self.window.blit(color_cell, self.get_surf_loc(self.locs[yeri]))
            #self.yerilistesi.append(self.s)
            self.yerilistesi[yerX] = (self.renkmavi,self.s)
        
        #Önceki vardiya mı normal vardiya mı durumuna göre renklendirme yapacak
        if (self.aracpernerden[yerX-1]==1):
            self.yerilistesi[self.stepsayac]=(self.renkkirmizi, self.encode(self.stepsayac,self.aracpersonel[self.stepsayac]-1))

        for eleman in self.yerilistesi:
            if eleman[1]>=0:
                color_cell = pygame.Surface(self.cell_size)
                color_cell.set_alpha(128)
                color_cell.fill(eleman[0]) #Mavi
                self.window.blit(color_cell, self.get_surf_loc(self.locs[eleman[1]]))

            
        
        color_cell = pygame.Surface(self.cell_size)
        color_cell.set_alpha(128)
        color_cell.fill(self.gezinti_color) #Beyaz
        self.window.blit(color_cell, self.get_surf_loc(self.locs[self.s]))
        
        
        print(self.aracpersonel)


        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            #if (taxi_row!=0 and taxi_col!=0 and pass_idx!=0 and dest_idx==0):
            #    print(self.ehliyetler)
            #    print(f"{self.s}- a = {pass_idx}, b = {taxi_col}, c= {taxi_row}, sec={dest_idx}")
            #    time.sleep(2) 
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        # map_loc[1] * 2 +1
        return (map_loc[1] * 1 +1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]
    
    def _render_text(self):
        #   C        B         A        SEC
        vardiya, taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
        if (taxi_row!=0 and taxi_col!=0 and pass_idx!=0 and dest_idx==0):
                print(self.ehliyetler)
                print(f"{self.s}- a = {pass_idx}, b = {taxi_col}, c= {taxi_row}, sec={dest_idx}")
                time.sleep(2) 
        #self.ii=self.ii+1
        #print(str(self.ii))

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
