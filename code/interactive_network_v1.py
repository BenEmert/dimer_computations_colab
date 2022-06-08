import itertools
import numpy as np
import pandas as pd

import param
import eqtk

import bokeh.plotting
import bokeh.io
import bokeh.models
import bokeh.palettes
import colorcet as cc
import panel as pn

from utilities import *

class network_model(param.Parameterized):
    
    n_monomers = param.Integer(3, bounds=(2, 6), precedence=-1)
    n_input = param.Integer(1, bounds=(1, 2), precedence=-1)
    c0_bounds = param.NumericTuple((-3, 3), precedence=-1)
    k_bounds = param.NumericTuple((-6, 6), precedence=-1)
    norm = param.Selector(['max', 'minmax', 'none'], 'max')
    n_titration = param.List([10], item_type=int, precedence=-1)
    out_weights = param.DataFrame()
        
    def __init__(self, seed = 42, **params):
        super().__init__(**params)   
        
        self.rng = np.random.default_rng(seed)
        self.solve_SS()
        self.update_output()

    @param.depends('n_monomers', 'n_input', watch=True, on_init=True)
    def default_network_params(self):
        #Define the fixed network variables
        self.n_species = number_of_species(self.n_monomers)        
        self.species_names = make_nXn_species_names(self.n_monomers)
        self.n_dimers = self.n_species - self.n_monomers
        self.dimer_names = self.species_names[self.n_monomers:]
        self.acc_names = self.species_names[self.n_input:self.n_monomers]
        self.n_acc = self.n_monomers - self.n_input
        self.Kij_names = make_Kij_names(n_input = self.n_input, n_accesory=(self.n_monomers-self.n_input))
        self.n_titration = self.n_titration*self.n_input
        self.rng = np.random.default_rng(42)
        self.out_weights = pd.DataFrame({'Dimer': self.dimer_names, 
                                        'Weight': [1.0]+[0.0]*(self.n_dimers-1)})

    @param.depends('n_monomers', watch=True, on_init=True)
    def update_N(self):
        self.N = make_nXn_stoich_matrix(self.n_monomers)
     
    @param.depends('n_titration', 'c0_bounds', watch=True, on_init=True)
    def default_C0(self):
        #consider allowing other monomers to be accessory using self.acc_index
        min_levels = [self.c0_bounds[0]] * self.n_monomers
        max_levels = [self.c0_bounds[1]] * self.n_monomers
        num_conc = self.n_titration + [1] * self.n_acc
        self.C0 = make_C0_grid(m=self.n_monomers, M0_min=min_levels, M0_max=max_levels, num_conc=num_conc)
        
        self.acc_c0_log = np.log10(self.C0[0,self.n_input:self.n_monomers])
    
    @param.depends('default_network_params', watch=True, on_init=True)
    def default_K(self):
        self.K = np.power(10, self.rng.uniform(self.k_bounds[0], self.k_bounds[1], self.n_dimers))
    
    @param.depends('default_K', watch=True, on_init=True)
    def make_network_cds(self):
        #make cds for vertices
        x,y = get_poly_vertices(self.n_monomers, r=1, start=0)
        #Set node size proportional to c0. 
        #For now, simply use log concentration (scaled)
        #Consider linear interpolation into logspace range i.e ~ np.interp(x, np.arange(c0_bounds), np.logspace(0, max_size))
        acc_norm = 5*(self.acc_c0_log-self.c0_bounds[0]+1)
        self.c0_norm = np.hstack(([20]*self.n_input, acc_norm))
        self.c0_log = ['input']*self.n_input+self.acc_c0_log.tolist()
        self.nodes_cds = bokeh.models.ColumnDataSource(dict(x=x, 
                                                            y=y, 
                                                            name=self.species_names[0:self.n_monomers],
                                                            c0_norm = self.c0_norm,
                                                            c0_log = self.c0_log))
        #make cds for edges
        connections = np.stack([list(i) for i in itertools.combinations_with_replacement(range(self.n_monomers), 2)])
        names_array = np.array(self.species_names)
        
        xs = [0] * connections.shape[0]
        ys = [0] * connections.shape[0]
        x2, y2 = get_poly_vertices(self.n_monomers, r = 1.4, start=0)
        
        for i,j in enumerate(connections):
            if j[0] == j[1]:
                circumference = pointsInCircum(x2[j[0]], y2[j[0]], r = 0.4)
                xs[i], ys[i] = circumference[:,0], circumference[:,1]
            else:
                edge = interp_points(x[j], y[j], n=5)
                xs[i], ys[i] = edge[:,0], edge[:,1]
    
        self.log_kij = np.round(np.log10(self.K), 2)
        self.kij_norm = 0.25*(self.log_kij-self.k_bounds[0]+1)
        self.edges_cds = bokeh.models.ColumnDataSource(dict(xs=xs, ys=ys, 
                                                            name = self.Kij_names,
                                                            log_kij = self.log_kij, 
                                                            kij_norm = self.kij_norm,
                                                            start = names_array[connections[:,0]],
                                                            end = names_array[connections[:,1]]))
        
   
    def solve_SS(self):
        self.S = eqtk.solve(c0=self.C0, N=self.N, K=self.K)
        
    
    @param.depends('norm', watch=True, on_init=False)
    def update_output(self):
        self.out_all = self.S[:,self.n_monomers:]
        # self.out_dot = np.matmul(self.out_all, self.out_weights['Weight']) #not using this for now
        if self.norm == "max":
            self.out_all = self.out_all/self.out_all.max(axis=0)
            # self.out_dot = self.out_dot/self.out_dot.max()  #not using this for now
        elif self.norm == "minmax":
            self.out_all = self.out_all - self.out_all.min(axis=0)
            self.out_all = self.out_all/self.out_all.max(axis=0)
            # self.out_dot = self.out_dot - self.out_dot.min() #not using this for now
            # self.out_dot = self.out_dot/self.out_dot.max()  #not using this for now

class accessory_monomer(param.Parameterized):
    index = param.Integer(bounds=(0, None), precedence=-1)
    value = param.Number(default=1, softbounds=(-3,3), step=0.05)
    model = param.ClassSelector(network_model, allow_None=True)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.param.name.precedence = -1
    
    @param.depends('value', watch=True, on_init=False)
    def update_C0(self):
        self.model.C0[:,self.index] = np.power(10.0, self.value)
    
    @param.depends('value', watch=True, on_init=False)
    def update_nodes(self):
        self.model.c0_log[self.index] = self.value
        self.model.c0_norm[self.index] = 5*(self.value-self.model.c0_bounds[0]+1.0)
                                                         
            

class Kij(param.Parameterized):
    index = param.Integer(bounds=(0, None), precedence=-1)
    value = param.Number(default=1, softbounds=(-6,6), step=0.05)
    model = param.ClassSelector(network_model, allow_None=True)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.param.name.precedence = -1
    
    @param.depends('value', watch=True, on_init=False)
    def update_K(self):
        self.model.K[self.index] = np.power(10.0, self.value)
    
    @param.depends('value', watch=True, on_init=False)
    def update_edges(self):
        self.model.log_kij[self.index] = self.value
        self.model.kij_norm[self.index] = 0.25*(self.value-self.model.k_bounds[0]+1.0)

class network_view_shared(param.Parameterized):
    
    xaxis_scale =  param.Selector(objects=['log', 'linear'])
    model = param.ClassSelector(network_model, allow_None=True)
    selector_trigger = param.Integer(0)
    slider_trigger = param.Integer(0)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.palette = cc.b_glasbey_hv
        # self.plot_output_1d()
        self.plot_network()
        self.make_widgets()
   
    @param.depends('model', watch=True, on_init=True)
    def make_model_params(self):
        self.acc_monomer_list = [accessory_monomer(name = self.model.acc_names[i], 
                                                   value = self.model.acc_c0_log[i],
                                                   index = self.model.n_input+i, 
                                                   model = self.model) for i in range(self.model.n_acc)]
        self.Kij_list = [Kij(name = self.model.Kij_names[i], 
                             value = self.model.log_kij[i], 
                             index = i, 
                             model = self.model) for i in range(self.model.n_dimers)]
        self.model_params = self.Kij_list + self.acc_monomer_list
        self.model_params_names = self.model.Kij_names+self.model.acc_names
        
    def plot_network(self):
        xpoints = np.hstack(self.model.edges_cds.data['xs'])
        ypoints = np.hstack(self.model.edges_cds.data['ys'])
        self.network_fig = bokeh.plotting.figure(
            frame_width=250,
            frame_height=250,
            x_axis_label=None,
            y_axis_label=None,
            toolbar_location = None,
            tools = '')

        #Format figure
        self.network_fig.axis.visible = False
        self.network_fig.grid.grid_line_color = None
        self.network_fig.outline_line_color = None
        #Add empty empty legend on right
        self.network_fig.add_layout(bokeh.models.Legend(title = 'Monomer', 
                                                        title_text_color = 'black', title_text_font_style = 'normal',
                                                        label_text_color = 'black', border_line_color = None), 
                                    'right')
        #Add renderers
        self.network_renderers_edges = self.network_fig.multi_line(xs="xs", ys="ys", 
                                                                   line_color="black", 
                                                                   line_width="kij_norm",
                                                                   source=self.model.edges_cds, 
                                                                   name = 'edges')
        
        self.model.nodes_cds.data['color'] = self.palette[self.model.n_dimers:self.model.n_species]
        self.network_renderers_nodes = self.network_fig.circle(x="x", y="y",
                                                               size = "c0_norm", 
                                                               fill_color = "color", 
                                                               line_color = "color", 
                                                               source=self.model.nodes_cds, 
                                                               name = 'points',
                                                               legend_group = 'name')
        
        #configure tooltips and add to figure
        edge_tooltip = [("Name", "@name"),("log Kij", "@log_kij")]
        node_tooltip = [("Monomer", "@name"), ("log c0", "@c0_log")]

        edge_hover = bokeh.models.HoverTool(tooltips=edge_tooltip, names = ['edges'])
        node_hover = bokeh.models.HoverTool(tooltips=node_tooltip, names = ['points'])

        self.network_fig.add_tools(edge_hover, node_hover)
        
    def make_widgets(self):
        self.param_Selector = pn.widgets.Select(options=dict(zip(self.model_params_names, self.model_params)), width=100, name='parameter:')
        # self.param_Selector.param.watch(self.update_slider, ['value'], onlychanged=True)
        # self.param_Selector.param.trigger('value')
        self.dimer_Selector = pn.widgets.Select(options=dict(zip(self.model.dimer_names, range(self.model.n_dimers))), width=100, name='Output dimer:')
        
        self.xaxis_Toggle = pn.widgets.RadioBoxGroup.from_param(self.param.xaxis_scale, inline = True)

        self.output_scale_Selector = pn.widgets.RadioBoxGroup.from_param(self.model.param.norm, inline = True)
        # self.output_scale_Selector.param.watch(self.out_scale_callback, ['value'], precedence = 1)
        
        self.hide_dimers_Toggle = pn.widgets.Toggle(name='hide dimer curves', button_type = 'default', width=100)
        # self.hide_dimers_Toggle.param.watch(self.toggle_dimers, ['value'])
    
        self.random_K_Button = pn.widgets.Button(name='randomize K', button_type = 'default', width=100)
        # self.random_K_Button.param.watch(self.randomize_K, ['value'])
    
    ####Update plots####
    
    def update_network_plot(self):
        #I'm not sure why this function is necessary for updating the plot
        #since the renderer's data source are automatically updated 
        #(they are pointing to objects that are updated by methods in the accessory_monomer and Kij classes
        #However, I'm only seeing the panel update when the renderers get updated as in this function
        #Note ,that I'm not using pn.io.push_notebook() since it is currently not supported on google colab. 
        self.network_renderers_edges.data_source.data['kij_norm'] = self.model.kij_norm
        self.network_renderers_edges.data_source.data['log_kij'] = self.model.log_kij
        self.network_renderers_nodes.data_source.data['c0_norm'] = self.model.c0_norm
        self.network_renderers_nodes.data_source.data['c0_log'] = self.model.c0_log
    
    ####View methods for Panel####
    @param.depends('slider_trigger')
    def network_view(self):
        return self.network_fig

class network_view_controller_1d(network_view_shared):
    def __init__(self, **params):
        super().__init__(**params)
        self.plot_output_1d()
        self.attach_callbacks()

    def plot_output_1d(self):
        self.output_1d_log = bokeh.plotting.figure(x_axis_type="log", 
                                                    plot_height=300, 
                                                    plot_width=325, 
                                                    x_axis_label='log10 [M1] total',
                                                    y_axis_label='Equilibrium output',
                                                    toolbar_location=None,
                                                    tools='pan,box_zoom,reset')
        #can't toggle axis scale with bokeh. Will create figure with both axis scales and toggle between them with callback
        self.output_1d_linear = bokeh.plotting.figure(x_axis_type="linear", 
                                                       plot_height=300, 
                                                       plot_width=325, 
                                                       x_axis_label='log10 [M1] total',
                                                       y_axis_label='Equilibrium output',
                                                       toolbar_location=None,
                                                       tools='pan,box_zoom,reset') 
        #Format figures
        self.format_fig(self.output_1d_log)
        self.format_fig(self.output_1d_linear)
        
        #Add renderers
        self.output_renderers_log = self.make_line_glyphs(self.output_1d_log)
        self.output_renderers_linear = self.make_line_glyphs(self.output_1d_linear)
        #Add legends
        self.make_out_legend(self.output_1d_log, self.output_renderers_log)
        self.make_out_legend(self.output_1d_linear, self.output_renderers_linear)
    
    def make_line_glyphs(self, fig):
        return [fig.line(self.model.C0[:,0], self.model.out_all[:,i], line_color = self.palette[i],
                         visible = True) for i in range(self.model.n_dimers)]
    
    def make_out_legend(self, fig, renderers):
        ncols = math.ceil(self.model.n_dimers/9) #For large n_monomers
        renderers_list = [[r] for r in renderers] #needed for bokeh.model.Legend
        dimer_names_split = np.array_split(self.model.dimer_names, ncols)
        renderers_split = np.array_split(renderers_list, ncols)
        for col in range(ncols):  
            legend = bokeh.models.Legend(items = list(zip(dimer_names_split[col].tolist(), renderers_split[col].tolist())),
                                        title = 'Dimer', title_text_color = 'black', title_text_font_style = 'normal',
                                        label_text_color = 'black', border_line_color = None, 
                                        click_policy = 'hide')
            fig.add_layout(legend, 'right')
            fig.plot_width += 75
    
    def format_fig(self, fig):
        fig.grid.grid_line_color = None
        fig.outline_line_color = None
        fig.axis.major_label_text_font_size = '16px'
        fig.axis.major_label_text_color = 'black'
        fig.axis.axis_label_text_color = 'black'
        fig.axis.axis_label_text_font_style = 'normal'
        fig.axis.axis_label_text_font_size = '16px'
        fig.y_range.update(start=0)
    
    def attach_callbacks(self):
        self.param_Selector.param.watch(self.update_slider, ['value'])
        self.param_Selector.param.trigger('value')
        
        self.output_scale_Selector.param.watch(self.out_scale_callback, ['value'])
        
        self.hide_dimers_Toggle.param.watch(self.toggle_dimers, ['value'])
    
        self.random_K_Button.param.watch(self.randomize_K, ['value'])
   
    ####Callbacks####    
    def update_slider(self, *events): #This is an ugly solution but it works...
        if hasattr(self, 'param_Slider_watcher'):
            self.param_Slider.param.unwatch(self.param_Slider_watcher)
        self.param_Slider = pn.widgets.FloatSlider.from_param(events[0].new)
        self.param_Slider.width = 150
        self.param_Slider.name = 'log10 value'
        self.param_Slider_watcher = self.param_Slider.param.watch(self.slider_callback, ['value'])
        self.param.trigger('selector_trigger')
        
    def slider_callback(self, *events):
        self.model.solve_SS()
        self.model.update_output()
        self.update_lines()
        self.update_network_plot()
        self.param.trigger('slider_trigger')
    
    def out_scale_callback(self, *events):
        #Unfortunately, when I try to simply update the renderers and the y_range on the figures 
        #occasionally, the y_ranges don't update (especially if xaxis_scale chanegd to 'linear')
        #workaround for now is to just remake the figs
        self.plot_output_1d() #
        self.param.trigger('xaxis_scale')
        
    def toggle_dimers(self, *events):
        for i in range(self.model.n_dimers):
            self.output_renderers_log[i].visible = events[0].old
            self.output_renderers_linear[i].visible = events[0].old
    
    def randomize_K(self, *events):
        new_vales = self.model.rng.uniform(self.model.k_bounds[0], self.model.k_bounds[1], self.model.n_dimers)
        for i, val in enumerate(new_vales):
            self.Kij_list[i].value = val
        if self.param_Selector.value.name[0] == 'M':
            self.slider_callback()
    
    ####Update plots####
    def update_lines(self):
        for i in range(self.model.n_dimers):
            self.output_renderers_log[i].data_source.data['y'] = self.model.out_all[:,i]
            self.output_renderers_linear[i].data_source.data['y'] = self.model.out_all[:,i]
    
    ####View methods for Panel####
    @param.depends('slider_trigger', 'xaxis_scale')
    def out_view(self):
        if self.xaxis_scale == 'log':
            return self.output_1d_log
        elif self.xaxis_scale == 'linear':
            return self.output_1d_linear
    
    @param.depends('selector_trigger')
    def widget_slider(self):
        return pn.Row(self.param_Slider)

    def panel(self):
        return pn.Row(self.out_view, self.network_view, pn.Column(self.param_Selector, self.widget_slider), 
                      pn.Column(pn.widgets.StaticText(value='x-axis scale:'), self.xaxis_Toggle, 
                                pn.widgets.StaticText(value='normalize output:'), self.output_scale_Selector,
                                self.hide_dimers_Toggle, self.random_K_Button))

class network_view_controller_2d(network_view_shared):
    def __init__(self, **params):
        super().__init__(**params)
        self.plot_output_2d()
        self.attach_callbacks()

    def plot_output_2d(self):
        self.output_2d = bokeh.plotting.figure(frame_width=250, frame_height=250,
                                               x_axis_label="log10 [M1] total", y_axis_label="log10 [M2] total",
                                               x_range=[0, self.model.n_titration[0]], 
                                               y_range=[0, self.model.n_titration[1]],
                                               toolbar_location=None)
        
        #Format figures
        self.format_fig(self.output_2d)
        
        #Make color_mapper
        self.color_mapper = bokeh.models.LinearColorMapper('Inferno256', low=0.0, high=self.model.out_all[:,self.dimer_Selector.value].max())
        #Add renderer
        self.output_renderer = self.make_heatmap_glyph(self.output_2d)
        
        #Add colorbar
        self.make_colorbar(self.output_2d)
        
        #Add tooltip
        self.output_2d.add_tools(bokeh.models.HoverTool(tooltips=[("value", "@image")]))
        
    def format_fig(self, fig):
        fig.grid.grid_line_color = None
        fig.outline_line_color = None
        fig.axis.major_label_text_font_size = '16px'
        fig.axis.major_label_text_color = 'black'
        fig.axis.axis_label_text_color = 'black'
        fig.axis.axis_label_text_font_style = 'normal'
        fig.axis.axis_label_text_font_size = '16px'
        xtick = [int(i) for i in np.linspace(0,self.model.n_titration[0],3)] #annoyingly, need a list of int for major_label_overrides to work
        ytick = [int(i) for i in np.linspace(0,self.model.n_titration[1],3)]
        xticklabels = [str(x) for x in np.linspace(self.model.c0_bounds[0], self.model.c0_bounds[1], 3)]
        yticklabels = [str(y) for y in np.linspace(self.model.c0_bounds[0], self.model.c0_bounds[1], 3)]
        fig.xaxis.ticker = xtick
        fig.yaxis.ticker = xtick
        fig.xaxis.major_label_overrides = dict(zip(xtick, xticklabels))
        fig.yaxis.major_label_overrides = dict(zip(ytick, yticklabels))
        
    def format_output_2d(self):
        return self.model.out_all[:,self.dimer_Selector.value].reshape(self.model.n_titration[0],-1).T
    
    def make_heatmap_glyph(self, fig):
        return fig.image(image=[self.format_output_2d()], x=0, y=0, dw=self.model.n_titration[0], dh=self.model.n_titration[1], color_mapper=self.color_mapper)
    
    def make_colorbar(self, fig):
        self.color_bar = bokeh.models.ColorBar(color_mapper=self.color_mapper)
        self.color_bar.major_label_text_color = 'black'
        self.color_bar.major_label_text_font_size = '14px'
        fig.add_layout(self.color_bar, 'right')
        
    def attach_callbacks(self):
        self.param_Selector.param.watch(self.update_slider, ['value'])
        self.param_Selector.param.trigger('value')
        
        self.dimer_Selector.param.watch(self.update_dimer, ['value'])
        self.output_scale_Selector.param.watch(self.out_scale_callback, ['value'])
        
        self.hide_dimers_Toggle.param.watch(self.toggle_dimers, ['value'])
    
        self.random_K_Button.param.watch(self.randomize_K, ['value'])
   
    ####Callbacks####    
    def update_slider(self, *events): #This is an ugly solution but it works...
        if hasattr(self, 'param_Slider_watcher'):
            self.param_Slider.param.unwatch(self.param_Slider_watcher)
        self.param_Slider = pn.widgets.FloatSlider.from_param(events[0].new)
        self.param_Slider.width = 150
        self.param_Slider.name = 'log10 value'
        self.param_Slider_watcher = self.param_Slider.param.watch(self.slider_callback, ['value'])
        self.param.trigger('selector_trigger')
        
    def slider_callback(self, *events):
        self.model.solve_SS()
        self.model.update_output()
        self.update_heatmap()
        self.update_network_plot()
        self.param.trigger('slider_trigger')
    
    def update_dimer(self, *events):
        self.update_heatmap()
        self.out_view()
        
    def out_scale_callback(self, *events):
        self.update_heatmap()
        self.out_view()
        
    def toggle_dimers(self, *events):
        for i in range(self.model.n_dimers):
            self.output_renderers_log[i].visible = events[0].old
            self.output_renderers_linear[i].visible = events[0].old
    
    def randomize_K(self, *events):
        new_vales = self.model.rng.uniform(self.model.k_bounds[0], self.model.k_bounds[1], self.model.n_dimers)
        for i, val in enumerate(new_vales):
            self.Kij_list[i].value = val
        if self.param_Selector.value.name[0] == 'M':
            self.slider_callback()
    
    ####Update plots####
    def update_heatmap(self):
        self.color_mapper.high = self.model.out_all[:,self.dimer_Selector.value].max()
        self.output_renderer.data_source.data['image'] = [self.format_output_2d()]
            
    ####View methods for Panel####
    @param.depends('slider_trigger')
    def out_view(self):
        return self.output_2d
    
    @param.depends('selector_trigger')
    def widget_slider(self):
        return pn.Row(self.param_Slider)

    def panel(self):
        return pn.Row(self.out_view, self.network_view, pn.Column(self.dimer_Selector, self.param_Selector, self.widget_slider), 
                      pn.Column(pn.widgets.StaticText(value ='normalize output:'), self.output_scale_Selector,self.random_K_Button))

